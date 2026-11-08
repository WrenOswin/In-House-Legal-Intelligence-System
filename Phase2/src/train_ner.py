"""
Phase-2 L-NER experiment driver: Phase-1 backbone + adaptation method + token-classification head,
trained on BIO tags over sliding windows, decoded with BIO-constrained Viterbi to character spans.

    from train_ner import run_experiment
    res = run_experiment(dict(exp="E1_A_full", backbone="A_d768", method="full", ...))

Everything a run produces goes to C:/kaggle/working/cs781/phase2/runs/<exp>/<run_id>/ and the
submission to Phase2/submissions/<exp>/submission_<stamp>_<run_id>/.
"""
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import load_backbone
from evaluate import strict_scores, error_breakdown, format_scores, write_submission
from ner_data import (Doc, IGNORE, N_TAGS, PROJECT_DIR, TAGS, centrality, load_split, tags_to_spans,
                      viterbi_bio, window_plan)
from peft import apply_method, count_params

WORK = Path("C:/kaggle/working/cs781/phase2")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULTS = dict(
    exp="E0", seed=1337,
    backbone="A_d768", bidirectional=True, max_len=None,     # None -> the backbone's context length
    method="full",                                            # full | partial | lora | lorma | adapter
    r=8, alpha=16, lora_dropout=0.05, n_train_layers=4, bottleneck=64, adapter_kind="pfeiffer",  # (only Pfeiffer is implemented)
    targets=("qkv", "q", "kv", "proj", "w12"),
    head_dropout=0.1,
    epochs=20, patience=5, micro_batch=4, grad_accum=2,
    lr=5e-5, head_lr=1e-3, weight_decay=0.01, warmup_frac=0.06, grad_clip=1.0,
    train_stride_frac=0.5,                                    # training windows overlap by 50% (+ random shift)
    infer_stride_frac=0.5,
    o_weight=1.0,                                             # class weight of tag O (<1 favours recall)
    split_punct=True,
    # knowledge distillation (student side)
    kd_teacher=None,                                          # path of a teacher run dir -> enables KD
    kd_alpha=0.5, kd_temp=2.0,
    full_data=False,                                          # True: train on train+val for `epochs` epochs, no early stop
    make_submission=True,
)

NOT_IN_RUN_ID = {"exp", "make_submission", "patience", "infer_stride_frac", "force"}


def run_id_of(cfg):
    keyed = {k: v for k, v in cfg.items() if k not in NOT_IN_RUN_ID}
    h = hashlib.sha1(json.dumps(keyed, sort_keys=True, default=str).encode()).hexdigest()[:6]
    tag = f"{cfg['backbone']}_{cfg['method']}" + ("_kd" if cfg["kd_teacher"] else "") + ("_all" if cfg["full_data"] else "")
    return f"{tag}_{h}"


class NERModel(nn.Module):
    def __init__(self, backbone, head_dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(backbone.d_model, N_TAGS)
        nn.init.normal_(self.head.weight, std=0.02); nn.init.zeros_(self.head.bias)

    def forward(self, idx):
        return self.head(self.drop(self.backbone(idx))).float()


def build_model(cfg):
    bb, tok, info = load_backbone(cfg["backbone"], cfg["bidirectional"])
    desc = apply_method(bb, cfg["method"], r=cfg["r"], alpha=cfg["alpha"], dropout=cfg["lora_dropout"],
                        n_train_layers=cfg["n_train_layers"], bottleneck=cfg["bottleneck"],
                        targets=tuple(cfg["targets"]))
    model = NERModel(bb, cfg["head_dropout"]).to(DEVICE)
    info["method_desc"] = desc
    info["max_len"] = cfg["max_len"] or bb.max_len
    return model, tok, info


# ----------------------------------------------------------------------------- inference
@torch.no_grad()
def predict_logprobs(model, doc, max_len, stride_frac=0.5, batch=8):
    """Log-probabilities [T, N_TAGS] for a whole document; every token keeps the window where it is most central."""
    model.eval()
    n = len(doc.ids)
    if n == 0:
        return np.zeros((0, N_TAGS), dtype=np.float32)
    plan = window_plan(n, max_len, stride_frac)
    out = np.zeros((n, N_TAGS), dtype=np.float32)
    best = np.full(n, -1, dtype=np.int64)
    for i in range(0, len(plan), batch):
        ws = plan[i:i + batch]
        x = torch.from_numpy(np.stack([doc.ids[s:s + bl] for s, bl in ws])).to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            lp = F.log_softmax(model(x), -1).cpu().numpy()
        for j, (s, bl) in enumerate(ws):
            c = centrality(s, bl, n)
            upd = c > best[s:s + bl]
            out[s:s + bl][upd] = lp[j][upd]
            best[s:s + bl][upd] = c[upd]
    return out


def decode_doc(doc, logp):
    skip = doc.spans[:, 1] == doc.spans[:, 0]
    return tags_to_spans(doc, viterbi_bio(logp, skip))


def predict_docs(model, docs, max_len, stride_frac=0.5):
    return {d.id: decode_doc(d, predict_logprobs(model, d, max_len, stride_frac)) for d in docs}


# ----------------------------------------------------------------------------- knowledge distillation
def teacher_soft_targets(teacher_dir, student_docs):
    """Teacher log-probs projected onto the STUDENT's tokens through character offsets, so teacher and
    student may use different tokenizers. A student token takes the distribution of the teacher token
    that contains its first character; if it starts inside that teacher token, B-X mass moves to I-X."""
    tcfg = json.load(open(Path(teacher_dir) / "config.json", encoding="utf-8"))
    teacher, ttok, tinfo = build_model(tcfg)
    teacher.load_state_dict(torch.load(Path(teacher_dir) / "model_best.pt", map_location=DEVICE))
    b2i = np.arange(N_TAGS)
    for k, t in enumerate(TAGS):
        if t.startswith("B-"):
            b2i[k] = TAGS.index("I-" + t[2:])
    targets = []
    for sd in student_docs:
        td = Doc(sd.id, sd.text, ttok, None, tcfg["split_punct"])
        tlp = predict_logprobs(teacher, td, tinfo["max_len"])
        tstart, tend = td.spans[:, 0], td.spans[:, 1]
        tgt = np.zeros((len(sd.ids), N_TAGS), dtype=np.float32)
        valid = np.zeros(len(sd.ids), dtype=bool)
        for t, (s, e) in enumerate(sd.spans):
            if e <= s:
                continue
            j = np.searchsorted(tend, s, side="right")          # first teacher token ending after s
            if j >= len(tstart) or tstart[j] > s:
                continue
            p = np.exp(tlp[j])
            if tstart[j] < s:                                    # student token starts inside the teacher token
                q = np.zeros_like(p); np.add.at(q, b2i, p); p = q
            tgt[t] = np.log(np.clip(p, 1e-8, 1)); valid[t] = True
        targets.append((tgt.astype(np.float16), valid))
    del teacher
    torch.cuda.empty_cache()
    return targets, tinfo


# ----------------------------------------------------------------------------- training
def run_experiment(user_cfg):
    cfg = {**DEFAULTS, **user_cfg}
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
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
    model, tok, info = build_model(cfg)
    L = info["max_len"]
    total, trainable = count_params(model)
    info.update(total_params=total, trainable_params=trainable)
    log(f"backbone {cfg['backbone']} ({info['model_class']}, arch {info['arch']}, {info['params']/1e6:.1f}M encoder params)"
        f" | {info['method_desc']} | bidirectional={model.backbone.bidirectional} | window {L}")
    log(f"params: total {total/1e6:.2f}M, trainable {trainable/1e6:.3f}M ({100*trainable/total:.2f}%)")

    texts, gold, test, split = load_split()
    train_ids = split["train"] + (split["val"] if cfg["full_data"] else [])
    val_ids = split["val"]
    t0 = time.time()
    train_docs = [Doc(k, texts[k], tok, gold[k], cfg["split_punct"]) for k in train_ids]
    val_docs = [Doc(k, texts[k], tok, gold[k], cfg["split_punct"]) for k in val_ids]
    for d in train_docs:                                       # whitespace-only tokens are not trained on
        d.labels[d.spans[:, 1] == d.spans[:, 0]] = IGNORE
    log(f"docs: train {len(train_docs)} ({sum(len(d.ids) for d in train_docs):,} tokens), val {len(val_docs)}"
        f" | tokenized in {time.time()-t0:.1f}s")

    kd = None
    if cfg["kd_teacher"]:
        t1 = time.time()
        kd, tinfo = teacher_soft_targets(cfg["kd_teacher"], train_docs)
        info["teacher"] = dict(dir=cfg["kd_teacher"], backbone=tinfo["name"], params=tinfo["params"])
        cover = np.mean(np.concatenate([v for _, v in kd]))
        log(f"KD teacher {tinfo['name']} ({tinfo['params']/1e6:.1f}M) soft targets ready in {time.time()-t1:.0f}s;"
            f" {100*cover:.1f}% of student tokens aligned; alpha={cfg['kd_alpha']} T={cfg['kd_temp']}")

    # ---- optimizer: the new head at head_lr; everything else that trains (backbone weights for full/partial,
    # LoRA/LoRMA/adapter/bias parameters for PEFT) at lr
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    groups = [{"params": list(model.head.parameters()), "lr": cfg["head_lr"], "weight_decay": 0.0},
              {"params": body, "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]}]
    groups = [g for g in groups if g["params"]]
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), fused=DEVICE == "cuda")

    w = torch.ones(N_TAGS, device=DEVICE); w[0] = cfg["o_weight"]

    def epoch_windows(rng):
        out = [(di, *win) for di, d in enumerate(train_docs)
               for win in window_plan(len(d.ids), L, cfg["train_stride_frac"], shift=rng.random())]
        rng.shuffle(out)
        return out

    def stack(chunk, get, pad):
        rows = [get(di)[s:s + bl] for di, s, bl in chunk]
        Lmax = max(len(r) for r in rows)                       # only docs shorter than L give shorter rows
        rows = [r if len(r) == Lmax else np.concatenate([r, np.full((Lmax - len(r),) + r.shape[1:], pad, r.dtype)])
                for r in rows]
        return torch.from_numpy(np.stack(rows)).to(DEVICE)

    rng = random.Random(cfg["seed"])
    steps_per_epoch = math.ceil(len(epoch_windows(random.Random(0))) / (cfg["micro_batch"] * cfg["grad_accum"]))
    total_steps = steps_per_epoch * cfg["epochs"]
    warm = max(1, int(cfg["warmup_frac"] * total_steps))

    def lr_mult(step):
        return (step + 1) / warm if step < warm else max(0.0, (total_steps - step) / max(1, total_steps - warm))

    log(f"{steps_per_epoch} steps/epoch x {cfg['epochs']} epochs (micro_batch {cfg['micro_batch']} x accum {cfg['grad_accum']})")
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best_f1, best_ep, bad, step, history = -1.0, 0, 0, 0, []
    t_train = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        wins = epoch_windows(rng)
        loss_sum, nb, te = 0.0, 0, time.time()
        for bi in range(0, len(wins), cfg["micro_batch"]):
            chunk = wins[bi:bi + cfg["micro_batch"]]
            x = stack(chunk, lambda di: train_docs[di].ids, 0)
            y = stack(chunk, lambda di: train_docs[di].labels, IGNORE)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                logits = model(x)
            loss = F.cross_entropy(logits.view(-1, N_TAGS), y.view(-1), weight=w, ignore_index=IGNORE)
            if kd is not None:
                tl = stack(chunk, lambda di: kd[di][0], 0.0).float()
                tv = stack(chunk, lambda di: kd[di][1], False)
                T = cfg["kd_temp"]
                pt = F.softmax(tl / T, -1)
                kl = (pt * (F.log_softmax(tl / T, -1) - F.log_softmax(logits / T, -1))).sum(-1)
                kl = (kl * tv).sum() / tv.sum().clamp(min=1) * T * T
                loss = cfg["kd_alpha"] * loss + (1 - cfg["kd_alpha"]) * kl
            (loss / cfg["grad_accum"]).backward()
            loss_sum += loss.item(); nb += 1
            if nb % cfg["grad_accum"] == 0 or bi + cfg["micro_batch"] >= len(wins):
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], cfg["grad_clip"])
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * lr_mult(step)
                opt.step(); opt.zero_grad(set_to_none=True); step += 1
        ep_time = time.time() - te
        rec = dict(epoch=ep, train_loss=loss_sum / max(1, nb), epoch_sec=round(ep_time, 1))
        if not cfg["full_data"]:
            vpred = predict_docs(model, val_docs, L, cfg["infer_stride_frac"])
            sc = strict_scores(vpred, {d.id: gold[d.id] for d in val_docs})
            rec.update(val_macro_f1=sc["macro_f1"], val_micro_f1=sc["micro_f1"])
            improved = sc["macro_f1"] > best_f1
            if improved:
                best_f1, best_ep, bad = sc["macro_f1"], ep, 0
                torch.save(model.state_dict(), run_dir / "model_best.pt")
            else:
                bad += 1
            log(f"epoch {ep:2d} | loss {rec['train_loss']:.4f} | val macro-F1 {sc['macro_f1']:.4f} micro {sc['micro_f1']:.4f}"
                f" | {ep_time:.0f}s" + ("  *best" if improved else ""))
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
                  method=cfg["method"], method_desc=info["method_desc"], bidirectional=model.backbone.bidirectional,
                  total_params=total, trainable_params=trainable, best_epoch=best_ep,
                  train_minutes=round(train_min, 1), peak_vram_gb=round(peak_gb, 2), teacher=info.get("teacher"))

    # ---- final validation report (skipped for full-data runs: val is part of training there)
    if not cfg["full_data"]:
        t1 = time.time()
        vpred = predict_docs(model, val_docs, L, cfg["infer_stride_frac"])
        result["val_infer_sec"] = round(time.time() - t1, 1)
        vgold = {d.id: gold[d.id] for d in val_docs}
        sc = strict_scores(vpred, vgold)
        result.update(val_macro_f1=sc["macro_f1"], val_macro_p=sc["macro_p"], val_macro_r=sc["macro_r"],
                      val_micro_f1=sc["micro_f1"])
        json.dump(dict(scores=sc, errors=error_breakdown(vpred, vgold)), open(run_dir / "val_metrics.json", "w",
                  encoding="utf-8"), indent=1)
        json.dump(vpred, open(run_dir / "val_predictions.json", "w", encoding="utf-8"))
        log(format_scores(sc))
        log("errors:", error_breakdown(vpred, vgold))

    # ---- test predictions -> submission
    if cfg["make_submission"]:
        test_docs = [Doc(k, test[k], tok, None, cfg["split_punct"]) for k in test]
        tpred = predict_docs(model, test_docs, L, cfg["infer_stride_frac"])
        stamp = time.strftime("%Y%m%d_%H%M%S")
        sub_dir = PROJECT_DIR / "submissions" / cfg["exp"] / f"submission_{stamp}_{rid}"
        z = write_submission(tpred, test, sub_dir, dict(result, config=cfg))
        result["submission"] = str(z)
        log(f"submission: {z} ({sum(len(v) for v in tpred.values())} entities in {len(tpred)} docs)")
    json.dump(result, open(run_dir / "result.json", "w", encoding="utf-8"), indent=1, default=str)
    log(json.dumps({k: v for k, v in result.items() if k != "teacher"}, default=str))
    log_f.close()
    del model
    torch.cuda.empty_cache()
    return result


def submission_from_run(run_dir):
    """Write a test submission from a finished run's model_best.pt (for runs trained with make_submission=False)."""
    run_dir = Path(run_dir)
    cfg = json.load(open(run_dir / "config.json", encoding="utf-8"))
    result = json.load(open(run_dir / "result.json", encoding="utf-8"))
    model, tok, info = build_model(cfg)
    model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=DEVICE))
    _, _, test, _ = load_split()
    test_docs = [Doc(k, test[k], tok, None, cfg["split_punct"]) for k in test]
    tpred = predict_docs(model, test_docs, info["max_len"], cfg["infer_stride_frac"])
    sub_dir = PROJECT_DIR / "submissions" / cfg["exp"] / f"submission_{time.strftime('%Y%m%d_%H%M%S')}_{result['run_id']}"
    z = write_submission(tpred, test, sub_dir, dict(result, config=cfg))
    result["submission"] = str(z)
    json.dump(result, open(run_dir / "result.json", "w", encoding="utf-8"), indent=1, default=str)
    print(f"submission: {z} ({sum(len(v) for v in tpred.values())} entities)")
    del model; torch.cuda.empty_cache()
    return z
