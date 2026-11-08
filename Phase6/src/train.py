"""
Phase-6 summarization training driver (Phase-5 driver adapted to generation), built to leave the laptop usable:
  * Throttle: after every GPU burst the process sleeps so that the GPU is busy only `duty` of the time
  * process priority BELOW_NORMAL, 2 CPU threads, VRAM capped at vram_fraction (OOM instead of shared-RAM spill)
  * micro-batches of <= token_budget tokens, activation checkpointing

    python run.py '{"exp": "S1_A_lorma", "backbone": "A_d768", "method": "lorma"}'

Input per doc (A/C/D): <|bos|> source <|sep|> summary[:max_tgt] (<|endoftext|> if the summary fits), source =
`context` selection (summ.context: select / head / headtail) of max_len - max_tgt - 3 tokens. B: the same
selected source cut into src_chunks chunks of src_chunk_len tokens, encoded separately (Fusion-in-Decoder).
Loss: token cross-entropy on the summary tokens.

Knowledge distillation (sequence-level, Kim & Rush 2016): kd_teacher = a finished teacher run dir whose
train_gen.json holds teacher-generated summaries of the train docs (teacher_gen.py). Each time a train doc is
drawn, the student's target is the teacher summary with probability kd_mix, else the reference summary.
Word-level KD on logits is not possible here: the 140M teacher (35k BPE) and the 58M students (32k BPE) use
different tokenizers.

Model selection: teacher-forced val loss (all VAL_SIZE val docs) every 1/evals_per_epoch epoch. At the end the
best checkpoint generates summaries for val (ROUGE-1/2/L, length, repetition) and test (submission).

Writes C:/kaggle/working/cs781/phase6/runs/<exp>/<run_id>/ (config, log, history, model_best.pt with the
trainable parameters only, val/test generations) and Phase6/submissions/<exp>/submission_<stamp>_<run_id>/.
"""
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import psutil
import torch

import summ
from seq2seq import adapt, load_summarizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULTS = dict(
    exp="S0", seed=1337, backbone="A_d768", method="lorma",
    r=16, alpha=32, lora_dropout=0.05, bottleneck=128,
    max_len=2048, rope_scale=None, max_tgt=512, context="select",
    src_chunk_len=512, src_chunks=3, dec_rope_scale=2.0,           # B only
    lr=1e-4, weight_decay=0.01, warmup_frac=0.05, grad_clip=1.0,
    epochs=3, docs_per_step=16, evals_per_epoch=2, patience=3,
    grad_ckpt=False, max_train_docs=None,
    kd_teacher=None, kd_mix=0.5,                                   # sequence-level KD (student side)
    # decoding (val + test generation)
    max_new=512, min_new=200, rep_penalty=1.0, no_repeat=4, gen_batch=16, val_gen_docs=300,
    # laptop comfort (not part of the run id)
    duty=0.6, token_budget=4096, vram_fraction=0.75, cpu_threads=2,
    debug=False, make_submission=True,
)
NOT_IN_ID = {"exp", "duty", "token_budget", "vram_fraction", "cpu_threads", "make_submission", "patience",
             "evals_per_epoch", "force", "max_new", "min_new", "rep_penalty", "no_repeat", "gen_batch",
             "val_gen_docs"}
GEN_KEYS = ("max_new", "min_new", "rep_penalty", "no_repeat")


class Throttle:
    """Keeps the GPU busy only `duty` of the wall time: after each burst, sleep busy * (1/duty - 1)."""
    def __init__(self, duty):
        self.duty, self.t = duty, time.perf_counter()

    def pause(self):
        if self.duty >= 1 or DEVICE != "cuda":
            return
        torch.cuda.synchronize()
        busy = time.perf_counter() - self.t
        time.sleep(min(2.0, busy * (1 / self.duty - 1)))
        self.t = time.perf_counter()


def be_nice(cfg):
    p = psutil.Process()
    try:
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    torch.set_num_threads(cfg["cpu_threads"])
    if DEVICE == "cuda" and cfg["vram_fraction"]:
        torch.cuda.set_per_process_memory_fraction(cfg["vram_fraction"])


def run_id_of(cfg):
    h = hashlib.sha1(json.dumps({k: v for k, v in cfg.items() if k not in NOT_IN_ID}, sort_keys=True,
                                default=str).encode()).hexdigest()[:6]
    return f"{cfg['backbone']}_{cfg['method']}" + ("_kd" if cfg["kd_teacher"] else "") + \
        ("_debug" if cfg["debug"] else "") + f"_{h}"


class Pipeline:
    """Model + tokenized data + source construction for one config."""
    def __init__(self, cfg):
        self.cfg = cfg
        model, tok, info = load_summarizer(cfg["backbone"], cfg["max_len"], cfg["rope_scale"], cfg["dec_rope_scale"])
        info["method_desc"] = adapt(model.lm, model.blocks, cfg["method"], r=cfg["r"], alpha=cfg["alpha"],
                                    dropout=cfg["lora_dropout"], bottleneck=cfg["bottleneck"], d_model=model.d_model)
        model.grad_ckpt = cfg["grad_ckpt"]
        self.model, self.tok, self.info = model.to(DEVICE), tok, info
        self.data = {s: summ.load_tokens(info["tokenizer"], s) for s in ("train", "val", "test")}
        self.sal = summ.salience_scores()
        self.src_budget = cfg["max_len"] - cfg["max_tgt"] - 3
        self._src = {}

    def source(self, split, i):
        key = (split, i)
        if key not in self._src:
            c = self.cfg
            if self.model.arch == "B":
                ctx = summ.context(self.data[split][i], c["src_chunk_len"] * c["src_chunks"], c["context"],
                                   self.sal[split][i])
                L = c["src_chunk_len"]
                self._src[key] = [ctx[k:k + L] for k in range(0, max(1, len(ctx)), L)] or [ctx]
            else:
                self._src[key] = summ.context(self.data[split][i], self.src_budget, c["context"], self.sal[split][i])
        return self._src[key]

    def target(self, ids):
        t = ids[:self.cfg["max_tgt"]]
        return t, len(ids) <= self.cfg["max_tgt"]

    def n_tokens(self, src, tgt):
        return (sum(len(c) for c in src) if self.model.arch == "B" else len(src)) + len(tgt) + 3

    def val_loss(self, idx, throttle=None):
        self.model.eval()
        tot, n = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            for mb in micro_batches([(self.source("val", i), *self.target(self.data["val"][i]["summary"]))
                                     for i in idx], self.cfg["token_budget"], self.n_tokens):
                loss, k = self.model.loss(mb)
                tot += loss.item() * k; n += k
                if throttle is not None:
                    throttle.pause()
        self.model.train()
        return tot / n

    def generate(self, split, idx, throttle=None, **gen):
        self.model.eval()
        g = {k: gen.get(k, self.cfg[k]) for k in GEN_KEYS}
        order = sorted(idx, key=lambda i: -self.n_tokens(self.source(split, i), []))   # similar lengths together
        out = {}
        bs = self.cfg["gen_batch"]
        for s in range(0, len(order), bs):
            ids = order[s:s + bs]
            res = self.model.generate([self.source(split, i) for i in ids], throttle=throttle, **g)
            for i, r in zip(ids, res):
                out[i] = self.tok.decode(r).strip()
        self.model.train()
        return [out[i] for i in idx]


def micro_batches(items, budget, n_tokens):
    """items: (src, tgt, done) -> lists of items with <= budget padded tokens (length-sorted)."""
    items = sorted(items, key=lambda x: n_tokens(x[0], x[1]))
    out, cur, cur_max = [], [], 0
    for it in items:
        n = n_tokens(it[0], it[1])
        if cur and max(cur_max, n) * (len(cur) + 1) > budget:
            out.append(cur); cur, cur_max = [], 0
        cur.append(it); cur_max = max(cur_max, n)
    if cur:
        out.append(cur)
    return out


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in names}


def load_pipeline(run_dir):
    run_dir = Path(run_dir)
    cfg = {**DEFAULTS, **json.load(open(run_dir / "config.json", encoding="utf-8"))}
    pipe = Pipeline(cfg)
    missing = pipe.model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=DEVICE), strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    return pipe


def score_texts(preds, refs):
    r = summ.rouge(preds, refs)
    r["words"] = float(np.mean([len(p.split()) for p in preds]))
    r["rep4"] = summ.repetition(preds, 4)
    return r


def run_experiment(user_cfg):
    cfg = {**DEFAULTS, **user_cfg}
    if cfg["debug"]:
        cfg.update(epochs=1, make_submission=False, val_gen_docs=16)
    be_nice(cfg)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    rid = run_id_of(cfg)
    run_dir = summ.WORK / "runs" / cfg["exp"] / rid
    if (run_dir / "result.json").exists() and not user_cfg.get("force"):
        print(f"{cfg['exp']}/{rid} already done"); return json.load(open(run_dir / "result.json", encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(run_dir / "config.json", "w", encoding="utf-8"), indent=1, default=str)
    log_f = open(run_dir / "train_log.txt", "a", encoding="utf-8")

    def log(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); log_f.write(s + "\n"); log_f.flush()

    throttle = Throttle(cfg["duty"])
    pipe = Pipeline(cfg)
    model, info = pipe.model, pipe.info
    rng = random.Random(cfg["seed"])
    train_idx = [i for i, d in enumerate(pipe.data["train"]) if d["summary"] is not None and len(d["summary"]) >= 20]
    val_idx = list(range(len(pipe.data["val"])))
    if cfg["max_train_docs"] and len(train_idx) > cfg["max_train_docs"]:
        train_idx = sorted(rng.sample(train_idx, cfg["max_train_docs"]))
    if cfg["debug"]:
        train_idx, val_idx = sorted(rng.sample(train_idx, 160)), val_idx[:32]
    gen_idx = val_idx[:cfg["val_gen_docs"]]
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"=== {cfg['exp']} / {rid} | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{cfg['backbone']} (arch {info['arch']}, {info['params']/1e6:.1f}M) | {info['method_desc']}")
    log(f"params: total {total/1e6:.2f}M, trainable {trainable/1e6:.3f}M | train {len(train_idx)} docs, val {len(val_idx)}"
        f" | max_len {cfg['max_len']} (rope x{info['rope_scale']}), src {pipe.src_budget if info['arch'] != 'B' else str(cfg['src_chunks']) + 'x' + str(cfg['src_chunk_len'])}"
        f" ({cfg['context']}), tgt {cfg['max_tgt']} | lr {cfg['lr']} | epochs {cfg['epochs']} | duty {cfg['duty']}")

    teacher = None
    if cfg["kd_teacher"]:
        tg = json.load(open(Path(cfg["kd_teacher"]) / "train_gen.json", encoding="utf-8"))
        by_id = {d["id"]: i for i, d in enumerate(pipe.data["train"])}
        teacher = {}
        for doc_id, text in tg.items():
            if doc_id in by_id and text.strip():
                teacher[by_id[doc_id]] = np.asarray(pipe.tok.encode_ordinary(text), dtype=np.int32)
        info["teacher"] = cfg["kd_teacher"]
        log(f"KD: {len(teacher)} teacher summaries from {cfg['kd_teacher']} | kd_mix {cfg['kd_mix']}")

    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"], fused=DEVICE == "cuda")
    spe = math.ceil(len(train_idx) / cfg["docs_per_step"]); total_steps = spe * cfg["epochs"]
    warm = max(1, int(cfg["warmup_frac"] * total_steps))
    eval_at = {round(spe * (k + 1) / cfg["evals_per_epoch"]) for k in range(cfg["evals_per_epoch"])}

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    v0 = pipe.val_loss(val_idx, throttle)
    log(f"val loss before training: {v0:.4f}")
    best, best_at, bad, step, hist, n_tok, stop = float("inf"), None, 0, 0, [dict(step=0, val_loss=v0)], 0, False
    t0 = time.time()
    model.train()
    for ep in range(1, cfg["epochs"] + 1):
        order = train_idx[:]; rng.shuffle(order)
        loss_sum, n_sum = 0.0, 0
        for s0 in range(0, len(order), cfg["docs_per_step"]):
            items = []
            for i in order[s0:s0 + cfg["docs_per_step"]]:
                ref = pipe.data["train"][i]["summary"]
                use_t = teacher is not None and i in teacher and rng.random() < cfg["kd_mix"]
                items.append((pipe.source("train", i), *pipe.target(teacher[i] if use_t else ref)))
            mbs = micro_batches(items, cfg["token_budget"], pipe.n_tokens)
            n_step = sum(len(t) + int(d) for _, t, d in items)
            for mb in mbs:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                    loss, k = model.loss(mb)
                (loss * k / n_step).backward()
                loss_sum += loss.item() * k; n_sum += k
                n_tok += sum(pipe.n_tokens(s, t) for s, t, _ in mb)
                throttle.pause()
            torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            mult = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total_steps - warm)))
            for gr in opt.param_groups:
                gr["lr"] = cfg["lr"] * mult
            opt.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 25 == 0:
                el = time.time() - t0
                log(f"  step {step}/{total_steps} | loss {loss_sum/max(1,n_sum):.4f} | {n_tok/el:.0f} tok/s | "
                    f"{el/60:.1f} min | peak {torch.cuda.max_memory_allocated()/2**30:.2f} GB")
            if step - (ep - 1) * spe in eval_at:
                vl = pipe.val_loss(val_idx, throttle)
                rec = dict(epoch=ep, step=step, train_loss=round(loss_sum / max(1, n_sum), 4), val_loss=round(vl, 4),
                           minutes=round((time.time() - t0) / 60, 1))
                hist.append(rec)
                json.dump(hist, open(run_dir / "history.json", "w", encoding="utf-8"), indent=1)
                better = vl < best
                if better:
                    best, best_at, bad = vl, (ep, step), 0
                    torch.save(trainable_state(model), run_dir / "model_best.pt")
                else:
                    bad += 1
                log(f"epoch {ep} step {step} | train loss {rec['train_loss']:.4f} | val loss {vl:.4f}"
                    + ("  *best" if better else ""))
                if bad >= cfg["patience"]:
                    log("early stop"); stop = True; break
        if stop:
            break
    train_min = (time.time() - t0) / 60
    peak = torch.cuda.max_memory_allocated() / 2**30 if DEVICE == "cuda" else 0.0
    model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=DEVICE), strict=False)

    t1 = time.time()
    val_pred = pipe.generate("val", gen_idx, throttle)
    val_sec = time.time() - t1
    splits = summ.build_splits()
    ref_txt = [splits["val"][i]["summary"] for i in gen_idx]
    sc = score_texts(val_pred, ref_txt)
    res = dict(exp=cfg["exp"], run_id=rid, backbone=cfg["backbone"], method=cfg["method"], method_desc=info["method_desc"],
               total_params=total, trainable_params=trainable, best_at=best_at, best_val_loss=best,
               train_minutes=round(train_min, 1), tok_per_s=round(n_tok / max(1, train_min * 60)),
               peak_vram_gb=round(peak, 2), val_gen_sec=round(val_sec, 1), val_gen_docs=len(gen_idx),
               decoding={k: cfg[k] for k in GEN_KEYS}, val_scores=sc, teacher=info.get("teacher"), duty=cfg["duty"])
    log(f"val generation ({len(gen_idx)} docs, {val_sec:.0f}s): {json.dumps(sc)}")
    json.dump({splits["val"][i]["id"]: p for i, p in zip(gen_idx, val_pred)},
              open(run_dir / "val_gen.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    if cfg["make_submission"]:
        t1 = time.time()
        te_idx = list(range(len(pipe.data["test"])))
        te_pred = pipe.generate("test", te_idx, throttle)
        res["test_gen_sec"] = round(time.time() - t1, 1)
        res["test_words"] = float(np.mean([len(p.split()) for p in te_pred]))
        by_id = {pipe.data["test"][i]["id"]: p for i, p in zip(te_idx, te_pred)}
        json.dump(by_id, open(run_dir / "test_gen.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
        sub = summ.PROJECT / "submissions" / cfg["exp"] / f"submission_{time.strftime('%Y%m%d_%H%M%S')}_{rid}"
        res["submission"] = str(summ.write_submission(by_id, sub, dict(res, config=cfg)))
        log(f"submission: {res['submission']} (test {res['test_words']:.0f} words/summary)")
    json.dump(res, open(run_dir / "result.json", "w", encoding="utf-8"), indent=1, default=str)
    log(json.dumps(res, default=str))
    log_f.close()
    del pipe, model
    torch.cuda.empty_cache()
    return res
