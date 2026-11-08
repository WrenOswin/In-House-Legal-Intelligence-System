"""
Phase-4 CJPE training driver, built to leave the laptop usable while it runs:
  * Throttle: after every GPU burst (one micro-batch forward/backward) the process sleeps so that the GPU is
    busy only `duty` of the time (default 0.6), and Windows / the desktop get the GPU in between
  * process priority BELOW_NORMAL, 2 CPU threads, VRAM capped at vram_fraction (OOM instead of shared-RAM spill)
  * tokens memory-mapped (cjpe.Tokens), small micro-batches (token_budget 2048), <= 4 chunks per training doc

    python run.py '{"exp": "J1_A_lorma", "backbone": "A_d768", "method": "lorma"}'

Writes C:/kaggle/working/cs781/phase4/runs/<exp>/<run_id>/ (config, log, history, model_best.pt, val/test
logits) and Phase4/submissions/<exp>/submission_<stamp>_<run_id>/. Selection and the decision threshold use
only val (994 clean dev docs).
"""
import hashlib
import json
import math
import pickle
import random
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F

import cjpe
from backbone import load_backbone
from model import DocModel
from peft import apply_method, count_params

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULTS = dict(
    exp="J0", seed=1337, backbone="A_d768", method="lorma",
    r=8, alpha=16, lora_dropout=0.05, n_train_layers=4, bottleneck=64,
    chunk_len=512, train_chunks=4, eval_chunks=8,             # chunks per doc: random in training, even at eval
    lr=1e-4, head_lr=1e-3, weight_decay=0.01, warmup_frac=0.06, grad_clip=1.0,
    epochs=2, docs_per_step=32, evals_per_epoch=2, patience=3,
    class_weight=True, grad_ckpt=False,
    max_train_docs=None, years=None,                          # optional train subsets (lighter / era-matched)
    kd_teacher=None, kd_alpha=0.5, kd_temp=2.0,               # knowledge distillation (student side)
    # laptop comfort (not part of the run id)
    duty=0.6, token_budget=2048, vram_fraction=0.6, cpu_threads=2,
    debug=False, make_submission=True,
)
NOT_IN_ID = {"exp", "duty", "token_budget", "vram_fraction", "cpu_threads", "make_submission", "patience",
             "evals_per_epoch", "eval_chunks", "force"}


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
    def __init__(self, cfg):
        self.cfg = cfg
        bb, tok, info = load_backbone(cfg["backbone"], bidirectional=False)   # A/D stay causal as pretrained
        info["method_desc"] = apply_method(bb, cfg["method"], r=cfg["r"], alpha=cfg["alpha"],
                                           dropout=cfg["lora_dropout"], n_train_layers=cfg["n_train_layers"],
                                           bottleneck=cfg["bottleneck"])
        bb.grad_ckpt = cfg["grad_ckpt"]
        self.model = DocModel(bb).to(DEVICE)
        self.info, self.pad = info, tok.pad_id
        tn = Path(info["tokenizer"]).stem
        self.toks = {s: cjpe.Tokens(tn, s, tok) for s in cjpe.SPLITS}

    def doc(self, split, i, rng=None):
        n = self.cfg["train_chunks"] if rng is not None else self.cfg["eval_chunks"]
        return cjpe.chunks(self.toks[split][i], self.cfg["chunk_len"], n, rng)

    @torch.no_grad()
    def predict(self, split, idx, throttle=None, batch=8):
        self.model.eval()
        out = []
        for s in range(0, len(idx), batch):
            docs = [self.doc(split, i) for i in idx[s:s + batch]]
            out.append(self.model(docs, self.pad, DEVICE, self.cfg["token_budget"], throttle).cpu().numpy())
        return np.concatenate(out)


def load_pipeline(run_dir):
    run_dir = Path(run_dir)
    cfg = {**DEFAULTS, **json.load(open(run_dir / "config.json", encoding="utf-8"))}
    pipe = Pipeline(cfg)
    pipe.model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=DEVICE))
    return pipe


def teacher_logits(teacher_dir, train_idx, throttle):
    """Teacher logits for the student's training docs (document level -> any tokenizer). Cached in the
    teacher's run dir."""
    f = Path(teacher_dir) / "train_logits.npy"
    if f.exists():
        return np.load(f)
    pipe = load_pipeline(teacher_dir)
    lg = pipe.predict("train", list(range(len(pipe.toks["train"]))), throttle)
    np.save(f, lg)
    del pipe
    torch.cuda.empty_cache()
    return lg


def run_experiment(user_cfg):
    cfg = {**DEFAULTS, **user_cfg}
    if cfg["debug"]:
        cfg.update(epochs=1, make_submission=False)
    be_nice(cfg)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    rid = run_id_of(cfg)
    run_dir = cjpe.WORK / "runs" / cfg["exp"] / rid
    if (run_dir / "result.json").exists() and not user_cfg.get("force"):
        print(f"{cfg['exp']}/{rid} already done"); return json.load(open(run_dir / "result.json", encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(run_dir / "config.json", "w", encoding="utf-8"), indent=1, default=str)
    log_f = open(run_dir / "train_log.txt", "a", encoding="utf-8")

    def log(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); log_f.write(s + "\n"); log_f.flush()

    throttle = Throttle(cfg["duty"])
    m = cjpe.meta()
    pipe = Pipeline(cfg)
    model, info = pipe.model, pipe.info
    rng = random.Random(cfg["seed"])
    ytr = np.array(m["train"]["labels"]); yva = np.array(m["val"]["labels"])
    train_idx = list(range(len(ytr)))
    if cfg["years"]:
        train_idx = [i for i in train_idx if cfg["years"][0] <= m["train"]["years"][i] <= cfg["years"][1]]
    if cfg["max_train_docs"] and len(train_idx) > cfg["max_train_docs"]:
        train_idx = sorted(rng.sample(train_idx, cfg["max_train_docs"]))
    val_idx = list(range(len(yva)))
    if cfg["debug"]:
        train_idx, val_idx = rng.sample(train_idx, 64), val_idx[:16] + val_idx[-16:]
    total, trainable = count_params(model)
    head_n = sum(p.numel() for p in model.head.parameters())
    cnt = np.bincount(ytr[train_idx], minlength=2)
    log(f"=== {cfg['exp']} / {rid} | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{cfg['backbone']} ({info['model_class']}, {info['params']/1e6:.1f}M) | {info['method_desc']}")
    log(f"params: total {total/1e6:.2f}M, trainable {trainable/1e6:.3f}M = backbone {(trainable-head_n)/1e6:.3f}M"
        f" + head {head_n/1e6:.3f}M | train {len(train_idx)} docs (0/1 {cnt[0]}/{cnt[1]}), val {len(val_idx)}")
    log(f"chunks {cfg['chunk_len']} x <= {cfg['train_chunks']} train / {cfg['eval_chunks']} eval | duty {cfg['duty']}"
        f" | token_budget {cfg['token_budget']} | lr {cfg['lr']} head {cfg['head_lr']}")

    kd = None
    if cfg["kd_teacher"]:
        t1 = time.time()
        kd = torch.from_numpy(teacher_logits(cfg["kd_teacher"], train_idx, throttle)).float()
        info["teacher"] = cfg["kd_teacher"]
        log(f"KD teacher logits ready ({time.time()-t1:.0f}s) alpha {cfg['kd_alpha']} T {cfg['kd_temp']}")

    cw = torch.tensor(cnt.sum() / (2.0 * np.maximum(cnt, 1)), dtype=torch.float32, device=DEVICE) \
        if cfg["class_weight"] else None
    head_p = list(model.head.parameters()); hid = {id(p) for p in head_p}
    body_p = [p for p in model.parameters() if p.requires_grad and id(p) not in hid]
    groups = [g for g in ({"params": head_p, "lr": cfg["head_lr"], "weight_decay": 0.0},
                          {"params": body_p, "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]}) if g["params"]]
    for g in groups:
        g["base"] = g["lr"]
    opt = torch.optim.AdamW(groups, fused=DEVICE == "cuda")
    params = [p for g in groups for p in g["params"]]
    spe = math.ceil(len(train_idx) / cfg["docs_per_step"]); total_steps = spe * cfg["epochs"]
    warm = max(1, int(cfg["warmup_frac"] * total_steps))
    eval_at = {round(spe * (k + 1) / cfg["evals_per_epoch"]) for k in range(cfg["evals_per_epoch"])}

    def evaluate():
        lg = pipe.predict("val", val_idx, throttle)
        p1 = torch.softmax(torch.from_numpy(lg), -1)[:, 1].numpy()
        thr = cjpe.best_threshold(p1, yva[val_idx])
        model.train()
        return cjpe.scores(p1 >= 0.5, yva[val_idx]), thr, cjpe.scores(p1 >= thr, yva[val_idx]), lg

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best, best_at, bad, step, hist, n_tok, stop = -1.0, None, 0, 0, [], 0, False
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        order = train_idx[:]; rng.shuffle(order)
        loss_sum, nd = 0.0, 0
        for s0 in range(0, len(order), cfg["docs_per_step"]):
            ids = order[s0:s0 + cfg["docs_per_step"]]
            docs = [pipe.doc("train", i, rng) for i in ids]
            groups_, cur, cur_t = [], [], 0          # whole documents per micro-batch, <= token_budget tokens
            for k, (cs, _) in enumerate(docs):
                t = sum(len(c) for c in cs)
                if cur and cur_t + t > cfg["token_budget"]:
                    groups_.append(cur); cur, cur_t = [], 0
                cur.append(k); cur_t += t
            groups_.append(cur)
            for g in groups_:
                lg = model([docs[k] for k in g], pipe.pad, DEVICE, cfg["token_budget"]).float()
                y = torch.tensor([ytr[ids[k]] for k in g], device=DEVICE)
                loss = F.cross_entropy(lg, y, weight=cw, reduction="sum")
                if kd is not None:
                    T = cfg["kd_temp"]; tl = kd[[ids[k] for k in g]].to(DEVICE)
                    kl = F.kl_div(F.log_softmax(lg / T, -1), F.log_softmax(tl / T, -1), log_target=True,
                                  reduction="sum") * T * T
                    loss = cfg["kd_alpha"] * loss + (1 - cfg["kd_alpha"]) * kl
                (loss / len(ids)).backward()
                loss_sum += loss.item(); nd += len(g); n_tok += sum(len(c) for k in g for c in docs[k][0])
                throttle.pause()
            torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            mult = (step + 1) / warm if step < warm else max(0.0, (total_steps - step) / max(1, total_steps - warm))
            for gr in opt.param_groups:
                gr["lr"] = gr["base"] * mult
            opt.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 100 == 0:
                el = time.time() - t0
                log(f"  step {step}/{total_steps} | loss {loss_sum/max(1,nd):.4f} | {n_tok/el:.0f} tok/s | {el/60:.1f} min")
            if step - (ep - 1) * spe in eval_at:
                sc, thr, sct, _ = evaluate()
                rec = dict(epoch=ep, step=step, loss=round(loss_sum / max(1, nd), 4), f1_05=sc["macro_f1"],
                           pred_pos=sc["pred_pos"], thr=round(thr, 2), f1_thr=sct["macro_f1"],
                           minutes=round((time.time() - t0) / 60, 1))
                hist.append(rec)
                json.dump(hist, open(run_dir / "history.json", "w", encoding="utf-8"), indent=1)
                better = sct["macro_f1"] > best
                if better:
                    best, best_at, bad = sct["macro_f1"], (ep, step), 0
                    torch.save(model.state_dict(), run_dir / "model_best.pt")
                else:
                    bad += 1
                log(f"epoch {ep} step {step} | loss {rec['loss']:.4f} | val macro-F1 {sc['macro_f1']:.4f} @0.5 "
                    f"(pred pos {sc['pred_pos']:.2f}) | {sct['macro_f1']:.4f} @thr {thr:.2f}" + ("  *best" if better else ""))
                if bad >= cfg["patience"]:
                    log("early stop"); stop = True; break
        if stop:
            break
    train_min = (time.time() - t0) / 60
    peak = torch.cuda.max_memory_allocated() / 2**30 if DEVICE == "cuda" else 0.0
    model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=DEVICE))
    t1 = time.time()
    sc, thr, sct, val_lg = evaluate()
    res = dict(exp=cfg["exp"], run_id=rid, backbone=cfg["backbone"], method=cfg["method"],
               method_desc=info["method_desc"], total_params=total, trainable_params=trainable, head_params=head_n,
               best_at=best_at, train_minutes=round(train_min, 1), tok_per_s=round(n_tok / max(1, train_min * 60)),
               peak_vram_gb=round(peak, 2), val_infer_sec=round(time.time() - t1, 1), duty=cfg["duty"],
               val_f1_05=sc["macro_f1"], threshold=round(thr, 2), val_f1_thr=sct["macro_f1"], val_scores_thr=sct,
               teacher=info.get("teacher"))
    log(f"best val: {sc['macro_f1']:.4f} @0.5 | {sct['macro_f1']:.4f} @thr {thr:.2f} | {sct}")
    logits = {"val": val_lg}
    if cfg["make_submission"]:
        t1 = time.time()
        te_lg = pipe.predict("test", list(range(len(m["test"]["ids"]))), throttle)
        res["test_infer_sec"] = round(time.time() - t1, 1)
        logits["test"] = te_lg
        p1 = torch.softmax(torch.from_numpy(te_lg), -1)[:, 1].numpy()
        pred = {i: int(p >= thr) for i, p in zip(m["test"]["ids"], p1)}
        res["test_pred_pos"] = round(float(np.mean(list(pred.values()))), 4)
        sub = cjpe.PROJECT_DIR / "submissions" / cfg["exp"] / f"submission_{time.strftime('%Y%m%d_%H%M%S')}_{rid}"
        res["submission"] = str(cjpe.write_submission(pred, sub, dict(res, config=cfg)))
        log(f"submission: {res['submission']} (thr {thr:.2f}, test pred pos {res['test_pred_pos']:.3f})")
    pickle.dump(logits, open(run_dir / "logits.pkl", "wb"))
    json.dump(res, open(run_dir / "result.json", "w", encoding="utf-8"), indent=1, default=str)
    log(json.dumps({k: v for k, v in res.items() if k != "val_scores_thr"}, default=str))
    log_f.close()
    del pipe, model
    torch.cuda.empty_cache()
    return res
