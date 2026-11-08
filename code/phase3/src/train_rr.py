"""
Phase-3 Rhetorical Role experiment driver: Phase-1 backbone + adaptation method + hierarchical sentence
tagger (pooled sentence vectors -> BiGRU over the document -> CRF), trained on random crops of documents.

    from train_rr import run_experiment
    res = run_experiment(dict(exp="R1_A_lorma", backbone="A_d768", method="lorma"))

A run writes to C:/kaggle/working/cs781/phase3/runs/<exp>/<run_id>/ (config, log, history, model_best.pt,
dev metrics, dev/test label probabilities for ensembling) and the submission to
Phase3/submissions/<exp>/submission_<stamp>_<run_id>/.
"""
import hashlib
import json
import math
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from backbone import load_backbone
from evaluate import format_scores, rr_scores, write_submission
from heads import RRModel
from peft import apply_method, count_params
from rr_data import (DOMAINS, LABELS, N_LABELS, PROJECT_DIR, WORK, label_counts, label_transitions, load_rr,
                     sentence_rows, tokenize_docs, window_rows)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_LR = {"full": 5e-5, "partial": 1e-4, "lora": 5e-4, "rslora": 5e-4, "lorma": 5e-4, "adapter": 5e-4,
              "ia3": 3e-3, "prefix": 1e-3}

DEFAULTS = dict(
    exp="R0", seed=1337,
    backbone="A_d768", bidirectional=False,                  # A/D: keep the pretrained causal mask (Phase 2 obs. 12)
    method="lorma",                                          # full | partial | lora | rslora | lorma | ia3 | prefix | adapter
    r=8, alpha=16, lora_dropout=0.05, n_train_layers=4, bottleneck=64, n_prefix=16, prefix_hidden=512,
    targets=("qkv", "q", "kv", "proj", "w12"),
    # input / context
    context_mode="window",                                   # window | sentence
    max_len=None,                                            # window length; None -> the backbone's context
    max_sent_tok=128,                                        # longer sentences are truncated
    train_overlap=0.0, infer_overlap=0.5,                    # window overlap (fraction of tokens)
    pool="mean",                                             # mean | last
    # head
    context="bigru", gru_hidden=256, gru_layers=2, head_dropout=0.1,
    use_crf=True, crf_init=True,                             # CRF transitions start at log P(l_t | l_t-1) of train
    ce_aux=0.0,                                              # + ce_aux * (class-weighted) CE on the emissions
    class_weight="none",                                     # none | sqrt   (inverse-frequency ^ 0.5, capped at 10)
    domain_emb=True,
    # optimisation
    epochs=15, patience=4, chunk_sents=64, chunks_per_step=4,
    lr=None, head_lr=1e-3, weight_decay=0.01, warmup_frac=0.06, grad_clip=1.0,
    token_budget=4096, grad_ckpt=False,                      # padded tokens per backbone micro-batch
    vram_fraction=0.75,                                      # cap on this process's GPU memory: an OOM error instead of
                                                             # Windows spilling into shared RAM (froze/crashed the laptop)
    step_pause=0.2,                                          # seconds of idle GPU after each optimizer step (thermal relief)
    select_metric="macro_f1_present",
    # knowledge distillation (student side)
    kd_teacher=None, kd_alpha=0.5, kd_temp=2.0,
    full_data=False,                                         # train on train+dev for `epochs`, no early stop
    debug=False,                                             # 4 train / 2 dev docs, 1 epoch, no submission
    make_submission=True,
)

NOT_IN_RUN_ID = {"exp", "make_submission", "patience", "infer_overlap", "force", "token_budget", "vram_fraction",
                 "step_pause"}


def run_id_of(cfg):
    keyed = {k: v for k, v in cfg.items() if k not in NOT_IN_RUN_ID}
    h = hashlib.sha1(json.dumps(keyed, sort_keys=True, default=str).encode()).hexdigest()[:6]
    tag = (f"{cfg['backbone']}_{cfg['method']}_{cfg['context_mode'][0]}" + ("_kd" if cfg["kd_teacher"] else "")
           + ("_all" if cfg["full_data"] else "") + ("_debug" if cfg["debug"] else ""))
    return f"{tag}_{h}"


class Pipeline:
    """Model + tokenizer + the doc -> rows/spans conversion for one config."""
    def __init__(self, cfg, train_docs_for_init=None):
        self.cfg = cfg
        bb, tok, info = load_backbone(cfg["backbone"], cfg["bidirectional"])
        info["method_desc"] = apply_method(bb, cfg["method"], r=cfg["r"], alpha=cfg["alpha"],
                                           dropout=cfg["lora_dropout"], n_train_layers=cfg["n_train_layers"],
                                           bottleneck=cfg["bottleneck"], targets=tuple(cfg["targets"]),
                                           n_prefix=cfg["n_prefix"], prefix_hidden=cfg["prefix_hidden"])
        bb.grad_ckpt = cfg["grad_ckpt"]
        crf_init = (label_transitions(train_docs_for_init) if cfg["use_crf"] and cfg["crf_init"]
                    and train_docs_for_init else None)
        self.model = RRModel(bb, N_LABELS, len(DOMAINS), cfg["context"], cfg["gru_hidden"], cfg["gru_layers"],
                             cfg["head_dropout"], cfg["use_crf"], crf_init, cfg["pool"], cfg["domain_emb"]).to(DEVICE)
        self.tok, self.info = tok, info
        self.tok_name = Path(info["tokenizer"]).stem
        self.sep = np.array(tok.encode_ordinary("\n"), dtype=np.int32)
        self.pad = tok.pad_id
        self.L = cfg["max_len"] or bb.max_len
        self.causal = bb.arch in "AD" and not bb.bidirectional
        info["max_len"] = self.L

    def rows(self, doc, a, b, overlap):
        if self.cfg["context_mode"] == "sentence":
            return sentence_rows(doc.toks[a:b], self.sep, self.cfg["max_sent_tok"])
        return window_rows(doc.toks[a:b], self.sep, self.L, self.cfg["max_sent_tok"], overlap, self.causal)

    def emissions(self, doc, a, b, overlap):
        rows, spans = self.rows(doc, a, b, overlap)
        vecs = self.model.encode(rows, spans, self.pad, DEVICE, self.cfg["token_budget"])
        return self.model.emissions(vecs, doc.domain)

    @torch.no_grad()
    def predict(self, docs):
        """-> {doc_id: (labels [n], probs [n, K], emissions [n, K])} for whole documents."""
        self.model.eval()
        out = {}
        for d in docs:
            em = self.emissions(d, 0, len(d), self.cfg["infer_overlap"])
            lab, prob = self.model.decode(em)
            out[d.id] = (lab, prob.astype(np.float32), em.cpu().numpy())
        return out


def as_labels(pred):
    return {k: [LABELS[i] for i in v[0]] for k, v in pred.items()}


def load_pipeline(run_dir, docs_for_tokens=None):
    """Rebuild a finished run's model (for KD teachers, ensembling, re-decoding)."""
    run_dir = Path(run_dir)
    cfg = {**DEFAULTS, **json.load(open(run_dir / "config.json", encoding="utf-8"))}
    pipe = Pipeline(cfg)
    pipe.model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=DEVICE))
    if docs_for_tokens:
        tokenize_docs(docs_for_tokens, pipe.tok, pipe.tok_name)
    return pipe


def teacher_emissions(teacher_dir, docs):
    """Teacher emission scores for every sentence of the student's training docs. Sentence-level targets,
    so teacher and student may use different tokenizers (each re-tokenizes the same sentences)."""
    import copy
    tdocs = [copy.copy(d) for d in docs]
    pipe = load_pipeline(teacher_dir, tdocs)
    pred = pipe.predict(tdocs)
    info = dict(pipe.info, dir=str(teacher_dir))
    del pipe
    torch.cuda.empty_cache()
    return {k: v[2] for k, v in pred.items()}, info


def run_experiment(user_cfg):
    cfg = {**DEFAULTS, **user_cfg}
    if cfg["lr"] is None:
        cfg["lr"] = DEFAULT_LR[cfg["method"]]
    if cfg["debug"]:
        cfg.update(epochs=1, make_submission=False)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    if DEVICE == "cuda" and cfg["vram_fraction"]:
        torch.cuda.set_per_process_memory_fraction(cfg["vram_fraction"])
    rid = run_id_of(cfg)
    run_dir = WORK / "runs" / cfg["exp"] / rid
    if (run_dir / "result.json").exists() and not user_cfg.get("force"):
        print(f"{cfg['exp']}/{rid} already finished -> loading result.json (pass force=True to retrain)")
        return json.load(open(run_dir / "result.json", encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(run_dir / "config.json", "w", encoding="utf-8"), indent=1, default=str)
    log_f = open(run_dir / "train_log.txt", "a", encoding="utf-8")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True); log_f.write(s + "\n"); log_f.flush()

    log(f"=== {cfg['exp']} / {rid} | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    data = load_rr()
    train_docs = data["train"] + (data["dev"] if cfg["full_data"] else [])
    dev_docs, test_docs = data["dev"], data["test"]
    if cfg["debug"]:
        train_docs = train_docs[:2] + train_docs[40:42]
        dev_docs = dev_docs[:1] + dev_docs[5:6]
    domain_of = {d.id: DOMAINS[d.domain] for d in data["train"] + data["dev"] + data["test"]}

    pipe = Pipeline(cfg, train_docs)
    model, info = pipe.model, pipe.info
    t0 = time.time()
    tokenize_docs(train_docs + dev_docs + test_docs, pipe.tok, pipe.tok_name)
    total, trainable = count_params(model)
    head_params = sum(p.numel() for p in model.head_parameters())
    log(f"backbone {cfg['backbone']} ({info['model_class']}, arch {info['arch']}, {info['params']/1e6:.1f}M) | "
        f"{info['method_desc']} | causal={pipe.causal} | {cfg['context_mode']} mode, window {pipe.L}")
    log(f"head: pool={cfg['pool']} context={cfg['context']} crf={cfg['use_crf']} domain_emb={cfg['domain_emb']}")
    log(f"params: total {total/1e6:.2f}M, trainable {trainable/1e6:.3f}M ({100*trainable/total:.2f}%) "
        f"= backbone {(trainable-head_params)/1e6:.3f}M + head {head_params/1e6:.3f}M")
    log(f"docs: train {len(train_docs)} ({sum(len(d) for d in train_docs):,} sents), dev {len(dev_docs)}, "
        f"test {len(test_docs)} | tokenized in {time.time()-t0:.1f}s")

    kd = None
    if cfg["kd_teacher"]:
        t1 = time.time()
        kd, tinfo = teacher_emissions(cfg["kd_teacher"], train_docs)
        info["teacher"] = dict(dir=cfg["kd_teacher"], backbone=tinfo["name"], params=tinfo["params"])
        log(f"KD teacher {tinfo['name']} ({tinfo['params']/1e6:.1f}M) emissions ready in {time.time()-t1:.0f}s;"
            f" alpha={cfg['kd_alpha']} T={cfg['kd_temp']}")

    cw = None
    if cfg["class_weight"] == "sqrt":
        c = label_counts(train_docs).astype(float)
        cw = np.where(c > 0, np.minimum(10.0, np.sqrt(c.max() / np.maximum(c, 1))), 0.0)
        cw = torch.tensor(cw / cw[c > 0].mean(), dtype=torch.float32, device=DEVICE)

    head = model.head_parameters()
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head, "lr": cfg["head_lr"], "weight_decay": 0.0},
              {"params": body, "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]}]
    groups = [g for g in groups if g["params"]]
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), fused=DEVICE == "cuda")
    params = [p for g in groups for p in g["params"]]

    C = cfg["chunk_sents"]

    def epoch_chunks(rng):
        out = []
        for di, d in enumerate(train_docs):
            n = len(d)
            start = -rng.randrange(C) if n > C else 0
            for a in range(start, n, C):
                out.append((di, max(0, a), min(n, a + C)))
        rng.shuffle(out)
        return out

    rng = random.Random(cfg["seed"])
    steps_per_epoch = math.ceil(len(epoch_chunks(random.Random(0))) / cfg["chunks_per_step"])
    total_steps = steps_per_epoch * cfg["epochs"]
    warm = max(1, int(cfg["warmup_frac"] * total_steps))

    def lr_mult(step):
        return (step + 1) / warm if step < warm else max(0.0, (total_steps - step) / max(1, total_steps - warm))

    log(f"~{steps_per_epoch} steps/epoch x {cfg['epochs']} epochs ({C} sentences/chunk x {cfg['chunks_per_step']} chunks/step)"
        f" | lr {cfg['lr']} head_lr {cfg['head_lr']}")
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best, best_ep, bad, step, history = -1.0, 0, 0, 0, []
    t_train = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        chunks = epoch_chunks(rng)
        loss_sum, nb, te = 0.0, 0, time.time()
        for ci, (di, a, b) in enumerate(chunks):
            d = train_docs[di]
            em = pipe.emissions(d, a, b, cfg["train_overlap"])
            y = torch.from_numpy(d.labels[a:b]).to(DEVICE)
            loss = model.loss(em, y, cw, cfg["ce_aux"])
            if kd is not None:
                T = cfg["kd_temp"]
                t = torch.from_numpy(kd[d.id][a:b]).to(DEVICE).float()
                kl = F.kl_div(F.log_softmax(em / T, -1), F.log_softmax(t / T, -1), log_target=True,
                              reduction="batchmean") * T * T
                loss = cfg["kd_alpha"] * loss + (1 - cfg["kd_alpha"]) * kl
            (loss / cfg["chunks_per_step"]).backward()
            loss_sum += loss.item(); nb += 1
            if nb % cfg["chunks_per_step"] == 0 or ci == len(chunks) - 1:
                torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * lr_mult(step)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
                if cfg["step_pause"]:
                    torch.cuda.synchronize(); time.sleep(cfg["step_pause"])
        ep_time = time.time() - te
        rec = dict(epoch=ep, train_loss=loss_sum / max(1, nb), epoch_sec=round(ep_time, 1))
        if not cfg["full_data"]:
            dpred = pipe.predict(dev_docs)
            sc = rr_scores(as_labels(dpred), {d.id: [LABELS[i] for i in d.labels] for d in dev_docs}, domain_of)
            rec.update(dev_macro_f1_present=sc["macro_f1_present"], dev_macro_f1_all=sc["macro_f1_all"],
                       dev_acc=sc["accuracy"], **{f"dev_{dom}_f1": sc[dom]["macro_f1_present"] for dom in DOMAINS if dom in sc})
            improved = sc[cfg["select_metric"]] > best
            if improved:
                best, best_ep, bad = sc[cfg["select_metric"]], ep, 0
                torch.save(model.state_dict(), run_dir / "model_best.pt")
            else:
                bad += 1
            dom_s = " ".join(f"{dom} {sc[dom]['macro_f1_present']:.4f}" for dom in DOMAINS if dom in sc)
            log(f"epoch {ep:2d} | loss {rec['train_loss']:.4f} | dev macro-F1 {sc['macro_f1_present']:.4f} "
                f"(all-13 {sc['macro_f1_all']:.4f}) acc {sc['accuracy']:.4f} | {dom_s} | {ep_time:.0f}s"
                + ("  *best" if improved else ""))
        else:
            log(f"epoch {ep:2d} | loss {rec['train_loss']:.4f} | {ep_time:.0f}s")
        history.append(rec)
        json.dump(history, open(run_dir / "history.json", "w", encoding="utf-8"), indent=1)
        if not cfg["full_data"] and bad >= cfg["patience"]:
            log(f"early stop: no improvement for {cfg['patience']} epochs")
            break
    train_min = (time.time() - t_train) / 60
    peak_gb = torch.cuda.max_memory_allocated() / 2**30 if DEVICE == "cuda" else 0.0
    if cfg["full_data"]:
        torch.save(model.state_dict(), run_dir / "model_best.pt"); best_ep = cfg["epochs"]
    model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=DEVICE))

    result = dict(exp=cfg["exp"], run_id=rid, run_dir=str(run_dir), backbone=cfg["backbone"], arch=info["arch"],
                  method=cfg["method"], method_desc=info["method_desc"], context_mode=cfg["context_mode"],
                  context=cfg["context"], use_crf=cfg["use_crf"], causal=pipe.causal,
                  total_params=total, trainable_params=trainable, head_params=head_params, best_epoch=best_ep,
                  train_minutes=round(train_min, 1), peak_vram_gb=round(peak_gb, 2), teacher=info.get("teacher"))

    probs = {}
    if not cfg["full_data"]:
        t1 = time.time()
        dpred = pipe.predict(dev_docs)
        result["dev_infer_sec"] = round(time.time() - t1, 1)
        sc = rr_scores(as_labels(dpred), {d.id: [LABELS[i] for i in d.labels] for d in dev_docs}, domain_of)
        result.update(dev_macro_f1_present=sc["macro_f1_present"], dev_macro_f1_all=sc["macro_f1_all"],
                      dev_macro_p=sc["macro_p_present"], dev_macro_r=sc["macro_r_present"], dev_acc=sc["accuracy"],
                      **{f"dev_{dom}_macro_f1": sc[dom]["macro_f1_present"] for dom in DOMAINS if dom in sc})
        json.dump(sc, open(run_dir / "dev_metrics.json", "w", encoding="utf-8"), indent=1)
        json.dump(as_labels(dpred), open(run_dir / "dev_predictions.json", "w", encoding="utf-8"))
        probs["dev"] = {k: v[1] for k, v in dpred.items()}
        log(format_scores(sc))

    if cfg["make_submission"]:
        tpred = pipe.predict(test_docs)
        probs["test"] = {k: v[1] for k, v in tpred.items()}
        stamp = time.strftime("%Y%m%d_%H%M%S")
        sub_dir = PROJECT_DIR / "submissions" / cfg["exp"] / f"submission_{stamp}_{rid}"
        z = write_submission(as_labels(tpred), test_docs, sub_dir, dict(result, config=cfg))
        result["submission"] = str(z)
        log(f"submission: {z}")
    pickle.dump(probs, open(run_dir / "probs.pkl", "wb"))       # label probabilities, for ensembling
    json.dump(result, open(run_dir / "result.json", "w", encoding="utf-8"), indent=1, default=str)
    log(json.dumps({k: v for k, v in result.items() if k != "teacher"}, default=str))
    log_f.close()
    del pipe, model
    torch.cuda.empty_cache()
    return result
