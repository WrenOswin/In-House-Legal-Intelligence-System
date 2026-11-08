"""
Phase-7 PCR training driver (Phase-5 driver layout: throttled, below-normal priority, memory-mapped tokens).

    python run.py '{"exp": "P1_A_lorma", "backbone": "A_d768", "method": "lorma"}'

Model: model.ChunkEncoder (Phase-1 backbone + adaptation -> unit chunk vectors), scored by chunk-level maxsim
(or one doc vector, score="doc").

One training step (contrastive, bi-encoder):
  * B train queries with >= 1 gold candidate; for each: one random gold candidate + `n_hard` hard negatives
    drawn from the query's BM25 (word-trigram) top-`hard_top` non-gold candidates (the query itself excluded)
  * chunk selection (select_chunks): every candidate keeps its `train_chunks` chunks with the most BPE-trigram
    overlap with the query; the query keeps the train_chunks - 1 chunks that overlap most with any of its own
    candidates + 1 random chunk. Same rule for gold and hard-negative candidates, so it gives no label away
  * loss: multi-positive InfoNCE of each query over ALL candidates of the batch (in-batch negatives),
    -log sum_{gold} exp(s/tau) / sum_all exp(s/tau); a candidate that is a query's own document is masked
  * GradCache (Gao et al. 2021): chunk vectors are first computed without gradients, the loss gradient w.r.t.
    the vectors is taken, then the chunks are re-encoded in token_budget micro-batches with gradients and the
    cached vector gradients are back-propagated -> big batches in little memory (the backbones have dropout 0,
    so both passes are identical)
KD (student side): (1 - kd_alpha) * InfoNCE + kd_alpha * KL(teacher || student) over each query's in-batch
candidate distribution; teacher = the full-pool maxsim scores of a finished run (teacher_scores.npy, cached).

Evaluation: the whole dev pool is encoded (eval_chunks evenly spaced chunks per case) -> F1@7 of the model
alone, and of the fusion z(BM25) + w * z(model) with w chosen on dev (fusion_w grid). Checkpoint selection by
the fused dev F1@7. The test submission uses the fused score with the chosen w.

Writes C:/kaggle/working/cs781/phase7/runs/<exp>/<run_id>/ and Phase7/submissions/<exp>/submission_<stamp>_<run_id>/.
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
import torch.nn.functional as F

import pcr
from backbone import load_backbone
from model import ChunkEncoder, pad_sets, score, score_blocks
from peft import apply_method, count_params

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_LR = {"full": 3e-5, "partial": 5e-5, "lora": 1e-4, "rslora": 1e-4, "lorma": 1e-4, "ia3": 3e-3,
              "prefix": 1e-3, "adapter": 5e-4, "bitfit": 1e-3}
DEFAULTS = dict(
    exp="P0", seed=1337, backbone="A_d768", method="lorma",
    r=8, alpha=16, lora_dropout=0.0, n_train_layers=4, bottleneck=64, n_prefix=16, prefix_hidden=512,
    score="maxsim", tau=0.05, chunk_len=512, train_chunks=4, eval_chunks=16,
    queries_per_step=16, n_hard=1, hard_top=30, steps=400,
    lr=None, head_lr=1e-3, weight_decay=0.01, warmup_frac=0.06, grad_clip=1.0,
    eval_every=100, grad_ckpt=False,
    bm25=(1.2, 1.0, -3), fusion_w=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0),
    kd_teacher=None, kd_alpha=0.5, kd_temp=1.0,
    # laptop comfort (not part of the run id)
    duty=0.6, token_budget=4096, vram_fraction=0.75, cpu_threads=2,
    debug=False, make_submission=True,
)
NOT_IN_ID = {"exp", "duty", "token_budget", "vram_fraction", "cpu_threads", "make_submission", "eval_every",
             "eval_chunks", "fusion_w", "force"}


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


def zscore(s):
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-6)


class TrigramSets:
    """Per document: sorted unique BPE-trigram hashes of every chunk (lazy cache) -> chunk overlap counts."""
    def __init__(self, toks, L):
        self.toks, self.L, self.cache = toks, L, {}

    def get(self, i):
        if i not in self.cache:
            t = self.toks[i].astype(np.int64)
            starts = list(range(0, max(1, len(t) - self.L + 1), self.L))
            if starts and starts[-1] + self.L < len(t):
                starts.append(len(t) - self.L)
            sets = []
            for s in starts:
                c = t[s:s + self.L]
                sets.append(np.unique(c[:-2] * 1_000_003 ** 2 + c[1:-1] * 1_000_003 + c[2:]) if len(c) > 2
                            else np.zeros(0, np.int64))
            self.cache[i] = (starts, sets)
        return self.cache[i]


def overlap(a_sets, b_sets):
    """[len(a), len(b)] number of shared trigrams between chunks."""
    return np.array([[len(np.intersect1d(x, y, assume_unique=True)) for y in b_sets] for x in a_sets], np.float32)


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        bb, tok, info = load_backbone(cfg["backbone"], bidirectional=False)   # A/D stay causal as pretrained
        info["method_desc"] = apply_method(bb, cfg["method"], r=cfg["r"], alpha=cfg["alpha"],
                                           dropout=cfg["lora_dropout"], n_train_layers=cfg["n_train_layers"],
                                           bottleneck=cfg["bottleneck"], n_prefix=cfg["n_prefix"],
                                           prefix_hidden=cfg["prefix_hidden"])
        bb.grad_ckpt = cfg["grad_ckpt"]
        self.model = ChunkEncoder(bb).to(DEVICE)
        self.info, self.pad = info, tok.pad_id
        tn = Path(info["tokenizer"]).stem
        self.toks = {s: pcr.Tokens(tn, s, tok) for s in pcr.SPLITS}

    @torch.no_grad()
    def pool_sets(self, split, throttle=None):
        """Every case of a pool -> padded chunk-vector sets [n, C, d] + mask (eval_chunks evenly spaced)."""
        self.model.eval()
        toks = self.toks[split]
        rows, owner = [], []
        for i in range(len(toks)):
            cs, _ = pcr.chunks(toks[i], self.cfg["chunk_len"], self.cfg["eval_chunks"])
            rows += cs; owner += [i] * len(cs)
        vecs = torch.cat([self.model(rows[s:s + 256], self.pad, DEVICE, self.cfg["token_budget"], throttle)
                          for s in range(0, len(rows), 256)])
        return pad_sets(vecs, owner, len(toks))

    def pool_scores(self, split, throttle=None):
        sets, mask = self.pool_sets(split, throttle)
        q = torch.as_tensor(pcr.meta()[split]["query_idx"], device=DEVICE)
        with torch.no_grad():
            return score_blocks(sets[q], mask[q], sets, mask, self.cfg["score"]).float().cpu().numpy()


def load_pipeline(run_dir):
    run_dir = Path(run_dir)
    cfg = {**DEFAULTS, **json.load(open(run_dir / "config.json", encoding="utf-8"))}
    pipe = Pipeline(cfg)
    pipe.model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=DEVICE))
    return pipe


def teacher_scores(teacher_dir, throttle):
    """Teacher maxsim scores [n_train_queries, n_train_pool] (pool level -> any tokenizer). Cached."""
    f = Path(teacher_dir) / "teacher_scores.npy"
    if f.exists():
        return np.load(f)
    pipe = load_pipeline(teacher_dir)
    s = pipe.pool_scores("train", throttle)
    np.save(f, s)
    del pipe
    torch.cuda.empty_cache()
    return s


def fusion(neural, bm, split, ws):
    """-> (best w, its scores, {w: F1@7}) for z(bm) + w * z(neural)."""
    zb, zn = zscore(bm), zscore(neural)
    table = {w: pcr.evaluate(zb + w * zn, split)["f1"] for w in ws}
    best = max(table, key=lambda w: (table[w], -w))
    return best, table


def select_chunks(ts, q, cands, n, rng):
    """Chunk starts for a query and each of its candidates (see module doc). -> (q_starts, [c_starts])."""
    q_starts, q_sets = ts.get(q)
    q_votes = np.zeros(len(q_starts), np.float32)
    c_out = []
    for c in cands:
        c_starts, c_sets = ts.get(c)
        ov = overlap(q_sets, c_sets)                                       # [nq, nc]
        best_c = np.argsort(-ov.max(0), kind="stable")[:n]
        c_out.append(sorted(c_starts[j] for j in best_c))
        q_votes = np.maximum(q_votes, ov.max(1))
    k = min(n, len(q_starts))
    top = list(np.argsort(-q_votes, kind="stable")[:max(1, k - 1)])        # best-aligned query chunks
    rest = [j for j in range(len(q_starts)) if j not in top]
    if rest and len(top) < k:
        top += rng.sample(rest, k - len(top))                              # + one random chunk
    return sorted(q_starts[j] for j in top), c_out


def run_experiment(user_cfg):
    cfg = {**DEFAULTS, **user_cfg}
    if cfg["lr"] is None:
        cfg["lr"] = DEFAULT_LR[cfg["method"]]
    if cfg["debug"]:
        cfg.update(steps=20, eval_every=10, make_submission=False)
    be_nice(cfg)
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    rid = run_id_of(cfg)
    run_dir = pcr.WORK / "runs" / cfg["exp"] / rid
    if (run_dir / "result.json").exists() and not user_cfg.get("force"):
        print(f"{cfg['exp']}/{rid} already done"); return json.load(open(run_dir / "result.json", encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(run_dir / "config.json", "w", encoding="utf-8"), indent=1, default=str)
    log_f = open(run_dir / "train_log.txt", "a", encoding="utf-8")

    def log(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); log_f.write(s + "\n"); log_f.flush()

    throttle = Throttle(cfg["duty"])
    m = pcr.meta()
    tr = m["train"]
    train_q = [i for i, g in enumerate(tr["gold"]) if g]                  # query positions with >= 1 gold
    bm_tr = pcr.bm25_scores("train", *cfg["bm25"])
    bm_dev = pcr.bm25_scores("dev", *cfg["bm25"])
    hard = {}
    for qi in train_q:
        s = bm_tr[qi].copy(); s[tr["query_idx"][qi]] = -np.inf; s[tr["gold"][qi]] = -np.inf
        hard[qi] = np.argsort(-s)[:cfg["hard_top"]].tolist()
    pipe = Pipeline(cfg)
    model, info = pipe.model, pipe.info
    total, trainable = count_params(model)
    L = cfg["chunk_len"]
    ts = TrigramSets(pipe.toks["train"], L)
    log(f"=== {cfg['exp']} / {rid} | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{cfg['backbone']} ({info['model_class']}, {info['params']/1e6:.1f}M) | {info['method_desc']} | score {cfg['score']}"
        f" tau {cfg['tau']}")
    log(f"params: total {total/1e6:.2f}M, trainable {trainable/1e6:.3f}M (proj {model.proj.weight.numel()/1e6:.3f}M)"
        f" | {len(train_q)} train queries, {cfg['queries_per_step']}/step x (1 gold + {cfg['n_hard']} BM25 hard neg"
        f" from top {cfg['hard_top']}) | {cfg['steps']} steps | chunks {L} x {cfg['train_chunks']} train /"
        f" {cfg['eval_chunks']} eval | lr {cfg['lr']} proj {cfg['head_lr']} | duty {cfg['duty']}")
    bm_ref = pcr.evaluate(bm_dev, "dev")
    log(f"BM25 {cfg['bm25']} dev: {bm_ref}")

    kd = None
    if cfg["kd_teacher"]:
        t1 = time.time()
        kd = teacher_scores(cfg["kd_teacher"], throttle)
        assert kd.shape == (len(tr["query_idx"]), len(tr["doc_ids"]))
        info["teacher"] = cfg["kd_teacher"]
        log(f"KD teacher scores ready ({time.time()-t1:.0f}s) alpha {cfg['kd_alpha']} T {cfg['kd_temp']}")

    proj_p = list(model.proj.parameters()); pid = {id(p) for p in proj_p}
    body_p = [p for p in model.parameters() if p.requires_grad and id(p) not in pid]
    groups = [g for g in ({"params": proj_p, "lr": cfg["head_lr"], "weight_decay": 0.0},
                          {"params": body_p, "lr": cfg["lr"], "weight_decay": cfg["weight_decay"]}) if g["params"]]
    for g in groups:
        g["base"] = g["lr"]
    opt = torch.optim.AdamW(groups, fused=DEVICE == "cuda")
    params = [p for g in groups for p in g["params"]]
    total_steps = cfg["steps"]; warm = max(1, int(cfg["warmup_frac"] * total_steps))
    gold_sets = [set(g) for g in tr["gold"]]

    def evaluate():
        s = pipe.pool_scores("dev", throttle)
        alone = pcr.evaluate(s, "dev")
        w, table = fusion(s, bm_dev, "dev", cfg["fusion_w"])
        model.train()
        return alone, w, table, s

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rng = random.Random(cfg["seed"])
    best, best_at, hist, n_tok, loss_sum, nl = -1.0, None, [], 0, 0.0, 0
    order = []
    t0 = time.time()
    model.train()
    for step in range(1, total_steps + 1):
        if len(order) < cfg["queries_per_step"]:
            order += rng.sample(train_q, len(train_q))
        qs, order = order[:cfg["queries_per_step"]], order[cfg["queries_per_step"]:]
        # ---- batch: candidates + chunk choice
        cand, per_q = [], []
        for qi in qs:
            cs = [rng.choice(tr["gold"][qi])] + rng.sample(hard[qi], cfg["n_hard"])
            per_q.append(cs); cand += cs
        cand = list(dict.fromkeys(cand))
        cpos = {c: k for k, c in enumerate(cand)}
        c_starts = {}
        q_rows, q_owner, c_rows, c_owner = [], [], [], []
        toks = pipe.toks["train"]
        for k, qi in enumerate(qs):
            q_doc = tr["query_idx"][qi]
            qst, cst = select_chunks(ts, q_doc, per_q[k], cfg["train_chunks"], rng)
            t = toks[q_doc]
            q_rows += [t[s:s + L] for s in qst]; q_owner += [k] * len(qst)
            for c, st in zip(per_q[k], cst):
                c_starts.setdefault(c, st)
        for c in cand:
            t = toks[c]
            c_rows += [t[s:s + L] for s in c_starts[c]]; c_owner += [cpos[c]] * len(c_starts[c])
        rows = q_rows + c_rows
        nq = len(q_rows)
        labels = torch.zeros(len(qs), len(cand), device=DEVICE)
        self_mask = torch.zeros(len(qs), len(cand), dtype=torch.bool, device=DEVICE)
        for k, qi in enumerate(qs):
            for c in cand:
                labels[k, cpos[c]] = float(c in gold_sets[qi])
                self_mask[k, cpos[c]] = c == tr["query_idx"][qi]
        # ---- GradCache pass 1: vectors without gradients
        with torch.no_grad():
            vecs = model(rows, pipe.pad, DEVICE, cfg["token_budget"])
        throttle.pause()
        vecs = vecs.detach().requires_grad_(True)
        qs_, qm = pad_sets(vecs[:nq], q_owner, len(qs))
        cs_, cm = pad_sets(vecs[nq:], c_owner, len(cand))
        sim = score(qs_, qm, cs_, cm, cfg["score"]) / cfg["tau"]
        sim = sim.masked_fill(self_mask, -1e4)
        lse_all = torch.logsumexp(sim, 1)
        lse_pos = torch.logsumexp(sim.masked_fill(labels == 0, -1e4), 1)
        loss = (lse_all - lse_pos).mean()
        if kd is not None:
            tq = torch.tensor(kd[np.ix_(qs, cand)], device=DEVICE) / cfg["tau"] / cfg["kd_temp"]
            tq = tq.masked_fill(self_mask, -1e4)
            kl = F.kl_div(F.log_softmax(sim / cfg["kd_temp"], 1), F.log_softmax(tq, 1), log_target=True,
                          reduction="batchmean") * cfg["kd_temp"] ** 2
            loss = (1 - cfg["kd_alpha"]) * loss + cfg["kd_alpha"] * kl
        loss.backward()
        g = vecs.grad.detach()
        # ---- pass 2: re-encode with gradients in micro-batches, back-propagate the cached vector gradients
        order_r = sorted(range(len(rows)), key=lambda i: len(rows[i]))
        i = 0
        while i < len(order_r):
            j = i + 1
            while j < len(order_r) and len(rows[order_r[j]]) * (j + 1 - i) <= cfg["token_budget"]:
                j += 1
            idx = order_r[i:j]
            v = model([rows[k] for k in idx], pipe.pad, DEVICE, cfg["token_budget"])
            (v * g[idx]).sum().backward()
            throttle.pause()
            i = j
        n_tok += 2 * sum(len(r) for r in rows)
        loss_sum += loss.item(); nl += 1
        torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        mult = step / warm if step <= warm else max(0.0, (total_steps - step) / max(1, total_steps - warm))
        for gr in opt.param_groups:
            gr["lr"] = gr["base"] * mult
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % 20 == 0:
            el = time.time() - t0
            log(f"  step {step}/{total_steps} | loss {loss_sum/nl:.4f} | {n_tok/el:.0f} tok/s | {el/60:.1f} min")
            loss_sum, nl = 0.0, 0
        if step % cfg["eval_every"] == 0 or step == total_steps:
            alone, w, table, _ = evaluate()
            rec = dict(step=step, dev_alone=alone, fusion_w=w, dev_fused=table[w], fusion_table=table,
                       minutes=round((time.time() - t0) / 60, 1))
            hist.append(rec)
            json.dump(hist, open(run_dir / "history.json", "w", encoding="utf-8"), indent=1)
            better = table[w] > best
            if better:
                best, best_at = table[w], step
                torch.save(model.state_dict(), run_dir / "model_best.pt")
            log(f"step {step} | dev F1@7 model alone {alone['f1']:.4f} (R@20 {alone['r@20']:.4f}) | fused w={w}"
                f" {table[w]:.4f} (BM25 {bm_ref['f1']:.4f})" + ("  *best" if better else ""))
    train_min = (time.time() - t0) / 60
    peak = torch.cuda.max_memory_allocated() / 2**30 if DEVICE == "cuda" else 0.0
    model.load_state_dict(torch.load(run_dir / "model_best.pt", map_location=DEVICE))
    t1 = time.time()
    alone, w, table, dev_s = evaluate()
    dev_sec = time.time() - t1
    fused = pcr.evaluate(zscore(bm_dev) + w * zscore(dev_s), "dev")
    res = dict(exp=cfg["exp"], run_id=rid, backbone=cfg["backbone"], method=cfg["method"], score=cfg["score"],
               method_desc=info["method_desc"], total_params=total, trainable_params=trainable, best_at=best_at,
               train_minutes=round(train_min, 1), tok_per_s=round(n_tok / max(1, train_min * 60)),
               peak_vram_gb=round(peak, 2), dev_infer_sec=round(dev_sec, 1), duty=cfg["duty"],
               dev_bm25=bm_ref, dev_alone=alone, fusion_w=w, dev_fused=fused, fusion_table=table,
               teacher=info.get("teacher"))
    log(f"best (step {best_at}) dev: model alone {alone} | fused w={w}: {fused}")
    np.save(run_dir / "dev_scores.npy", dev_s)
    if cfg["make_submission"]:
        t1 = time.time()
        te_s = pipe.pool_scores("test", throttle)
        res["test_infer_sec"] = round(time.time() - t1, 1)
        np.save(run_dir / "test_scores.npy", te_s)
        bm_te = pcr.bm25_scores("test", *cfg["bm25"])
        top = pcr.rank(zscore(bm_te) + w * zscore(te_s), pcr.meta()["test"]["query_idx"])
        sub = pcr.PROJECT_DIR / "submissions" / cfg["exp"] / f"submission_{time.strftime('%Y%m%d_%H%M%S')}_{rid}"
        res["submission"] = str(pcr.write_submission(top, sub, dict(res, config=cfg)))
        log(f"submission: {res['submission']} (fused w={w})")
    json.dump(res, open(run_dir / "result.json", "w", encoding="utf-8"), indent=1, default=str)
    log(json.dumps({k: v for k, v in res.items() if k != "fusion_table"}, default=str))
    log_f.close()
    del pipe, model
    torch.cuda.empty_cache()
    return res
