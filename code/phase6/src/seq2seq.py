"""
Phase-6 summarizers on top of the Phase-1 LMs (weights loaded unchanged; this file re-implements only the
forward passes so that every architecture gets the same extras: RoPE position interpolation, logits only
on summary positions, activation checkpointing, Houlsby adapters, and batched KV-cache generation).

A, C, D  (one stack)   sequence = <|bos|> source <|sep|> summary <|endoftext|>, loss on the summary only
   A  causal everywhere (as pretrained)
   C  prefix-LM: source part bidirectional, summary causal; segment embedding 0 = source, 1 = summary
   D  causal; local layers sliding window (256), global layers full causal (as pretrained)
B  (encoder-decoder, Fusion-in-Decoder): every source chunk (<= 768 tokens) is encoded separately, the chunk
   encodings are concatenated into one memory and the decoder cross-attends to all of it
   ("hierarchical encoder-decoder representation" of a long document); decoder input <|bos|> summary.

RoPE position interpolation (Chen et al. 2023): positions are divided by rope_scale, so a 1024-token
pretrained model sees a 2048-token sequence at the angles it knows (A/C: 2048 = 1536 source + 512 summary).

Generation: greedy (or sampling) with batched, left-padded prompts and a preallocated KV cache; repetition
penalty on already generated tokens, no-repeat n-gram blocking, min/max new tokens.
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import models_a, models_b, models_c, models_d
from bpe import BPETokenizer
from peft import Adapter, LoRALinear, LoRMALinear, DEFAULT_TARGETS

PHASE1_RUNS = Path("C:/kaggle/working/cs781/runs")
BACKBONES = {
    "A_d768": "A_decoder_only/d768_L12_T1024_v35k_5fbdf6",
    "A_d560": "A_decoder_only/d560_L8_T1024_v32k_6613f2",
    "B_d512": "B_encoder_decoder/d512_E6D6_S768T256_v32k_05421b",
    "C_d560": "C_prefix_lm/d560_L8_T1024_v32k_6cce41",
    "D_d560": "D_efficient/d560_L8_T2048_v32k_05e1ef",
}
_CLASSES = {"DecoderOnlyLM": models_a.DecoderOnlyLM, "EncoderDecoderLM": models_b.EncoderDecoderLM,
            "PrefixLM": models_c.PrefixLM, "EfficientLM": models_d.EfficientLM}
_ARCH = {"DecoderOnlyLM": "A", "EncoderDecoderLM": "B", "PrefixLM": "C", "EfficientLM": "D"}


# ---------------------------------------------------------------------------------------------- RoPE
def rope_cs(pos, head_dim, theta, scale):
    """pos [B, T] (long) -> (cos, sin) [B, 1, T, hd/2]; positions divided by `scale` (interpolation)."""
    inv = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=pos.device).float() / head_dim)
    a = (pos.float() / scale)[..., None] * inv
    return a.cos()[:, None], a.sin()[:, None]


def apply_rope(x, cs):
    cos, sin = cs
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)


# ---------------------------------------------------------------------------------------------- KV cache
class Cache:
    """Preallocated keys/values per layer: [B, Hkv, S_max, hd]; `n` positions filled."""
    def __init__(self, n_layer):
        self.k, self.v, self.n = [None] * n_layer, [None] * n_layer, 0

    def update(self, li, k, v, s_max):
        if self.k[li] is None:
            B, H, _, hd = k.shape
            self.k[li] = k.new_zeros(B, H, s_max, hd)
            self.v[li] = v.new_zeros(B, H, s_max, hd)
        T = k.size(2)
        self.k[li][:, :, self.n:self.n + T] = k
        self.v[li][:, :, self.n:self.n + T] = v
        return self.k[li][:, :, :self.n + T], self.v[li][:, :, :self.n + T]


def _proj_qkv(attn, x, kv_src=None):
    """q, k, v [B, H(kv), T, hd] for every attention flavour of the four architectures."""
    B, T, _ = x.shape
    H, hd = attn.n_head, attn.head_dim
    Hkv = getattr(attn, "n_kv_head", H)
    if hasattr(attn, "qkv"):                                  # A, C, B self-attention: fused [q | k | v]
        q, k, v = attn.qkv(x).split([H * hd, Hkv * hd, Hkv * hd], dim=-1)
        S = T
    else:                                                     # D, B cross-attention: q and fused [k | v]
        src = x if kv_src is None else kv_src
        S = src.size(1)
        q = attn.q(x)
        k, v = attn.kv(src).split([Hkv * hd, Hkv * hd], dim=-1)
    q = q.view(B, T, H, hd).transpose(1, 2)
    k = k.view(B, S, Hkv, hd).transpose(1, 2)
    v = v.view(B, S, Hkv, hd).transpose(1, 2)
    return q, k, v


def self_attention(attn, x, cs, mask=None, causal=False, window=None, cache=None, li=0, s_max=0):
    B, T, _ = x.shape
    q, k, v = _proj_qkv(attn, x)
    q, k = apply_rope(q, cs), apply_rope(k, cs)
    if cache is not None:
        k, v = cache.update(li, k, v, s_max)
    H, Hkv = q.size(1), k.size(1)
    if Hkv != H:
        k = k.repeat_interleave(H // Hkv, dim=1)
        v = v.repeat_interleave(H // Hkv, dim=1)
    if mask is None and causal and window and T > window:
        y = models_d.sliding_window_attention(q, k, v, window)
    else:
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal and mask is None)
    return attn.proj(y.transpose(1, 2).reshape(B, T, -1))


def _with_adapter(blk, name, h):
    ad = getattr(blk, name, None)
    return h if ad is None else ad(h)


# ---------------------------------------------------------------------------------------------- adaptation
class Bottleneck(nn.Module):
    """Houlsby adapter: h + up(gelu(down(h))), up = 0 at init (starts as the identity)."""
    def __init__(self, d_model, bottleneck=64):
        super().__init__()
        self.down, self.up = nn.Linear(d_model, bottleneck), nn.Linear(bottleneck, d_model)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, h):
        return h + self.up(F.gelu(self.down(h)))


def _replace_linears(blocks, make, targets):
    n = 0
    for blk in blocks:
        for parent in blk.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear) and name in targets:
                    setattr(parent, name, make(child)); n += 1
    return n


def adapt(lm, blocks, method, r=8, alpha=16, dropout=0.0, bottleneck=64, targets=DEFAULT_TARGETS, d_model=None):
    """Freeze the Phase-1 LM and add the trainable parts in place. Returns a description."""
    if method == "full":
        for p in lm.parameters():
            p.requires_grad = True
        return "full fine-tuning"
    for p in lm.parameters():
        p.requires_grad = False
    if method in ("lora", "lorma"):
        cls = LoRMALinear if method == "lorma" else LoRALinear
        n = _replace_linears(blocks, lambda m: cls(m, r=r, alpha=alpha, dropout=dropout), targets)
        return f"{method}: r={r} alpha={alpha} on {n} linear layers ({','.join(targets)})"
    if method == "pfeiffer":                                  # one adapter after every FFN
        for blk in blocks:
            blk.ffn = Adapter(blk.ffn, d_model, bottleneck)
        return f"pfeiffer adapters (after FFN), bottleneck {bottleneck}, {len(blocks)} blocks"
    if method == "houlsby":                                   # adapters after attention AND FFN + norms trained
        for blk in blocks:
            blk.adapter_attn = Bottleneck(d_model, bottleneck)
            blk.ffn = Adapter(blk.ffn, d_model, bottleneck)
            for m in blk.modules():
                if isinstance(m, (models_a.RMSNorm, models_b.RMSNorm, models_c.RMSNorm, models_d.RMSNorm)):
                    for p in m.parameters():
                        p.requires_grad = True
        return f"houlsby adapters (after attention and FFN, norms trained), bottleneck {bottleneck}, {len(blocks)} blocks"
    raise ValueError(method)


# ---------------------------------------------------------------------------------------------- models
class Summarizer(nn.Module):
    def __init__(self, lm, arch, tok, max_len=2048, rope_scale=1.0, dec_rope_scale=1.0):
        super().__init__()
        self.lm, self.arch, self.tok = lm, arch, tok
        self.max_len, self.rope_scale, self.dec_rope_scale = max_len, rope_scale, dec_rope_scale
        self.grad_ckpt = False
        lm.drop = nn.Identity()
        a0 = (lm.enc_blocks if arch == "B" else lm.blocks)[0].attn
        self.head_dim, self.theta = a0.head_dim, 10000.0              # theta set by load_summarizer
        self.pad, self.bos, self.sep, self.eot = tok.pad_id, tok.bos_id, tok.sep_id, tok.eot_id
        self.banned = [tok.pad_id, tok.bos_id, tok.sep_id, tok.mask_id]

    @property
    def blocks(self):
        return list(self.lm.enc_blocks) + list(self.lm.dec_blocks) if self.arch == "B" else list(self.lm.blocks)

    @property
    def d_model(self):
        return self.lm.tok_emb.embedding_dim

    def _ckpt(self, fn, x):
        if self.grad_ckpt and self.training and torch.is_grad_enabled():
            return checkpoint(fn, x, use_reentrant=False)
        return fn(x)

    # ------------------------------------------------------------------ one-stack models (A, C, D)
    def _stack(self, x, cs, masks, cache=None, s_max=0):
        """masks: per-layer (mask, causal, window)."""
        for li, blk in enumerate(self.lm.blocks):
            m, causal, window = masks[li]

            def fn(x, blk=blk, m=m, causal=causal, window=window, li=li):
                h = self_attention(blk.attn, blk.norm1(x), cs, m, causal, window, cache, li, s_max)
                x = x + _with_adapter(blk, "adapter_attn", h)
                return x + blk.ffn(blk.norm2(x))
            x = self._ckpt(fn, x)
        if cache is not None:
            cache.n += x.size(1)
        return self.lm.norm_f(x)

    def _embed(self, idx, seg=None):
        x = self.lm.tok_emb(idx)
        if self.arch == "C" and self.lm.seg_emb is not None and seg is not None:
            x = x + self.lm.seg_emb(seg)
        return x

    def _train_masks(self, T, prefix_len, device):
        if self.arch == "C":
            pos = torch.arange(T, device=device)
            m = (pos[None, None, :] < prefix_len[:, None, None]) | (pos[None, :, None] >= pos[None, None, :])
            return [(m[:, None], False, None)] * len(self.lm.blocks)
        if self.arch == "D":
            return [(None, True, blk.attn.window) for blk in self.lm.blocks]
        return [(None, True, None)] * len(self.lm.blocks)

    def loss(self, batch):
        """batch: list of (src int array, tgt int array, tgt_complete bool) -> mean token CE, n tokens."""
        if self.arch == "B":
            return self._loss_b(batch)
        dev = self.lm.tok_emb.weight.device
        seqs, pls = [], []
        for src, tgt, done in batch:
            s = [self.bos] + list(src) + [self.sep]
            pls.append(len(s))
            seqs.append(s + list(tgt) + ([self.eot] if done else []))
        T = max(map(len, seqs))
        idx = torch.full((len(seqs), T), self.pad, dtype=torch.long)
        for b, s in enumerate(seqs):
            idx[b, :len(s)] = torch.tensor(s)
        idx = idx.to(dev)
        prefix_len = torch.tensor(pls, device=dev)
        pos = torch.arange(T, device=dev)[None].expand(len(seqs), T)
        seg = (pos >= prefix_len[:, None]).long()
        cs = rope_cs(pos[:1], self.head_dim, self.theta, self.rope_scale)
        h = self._stack(self._embed(idx, seg), cs, self._train_masks(T, prefix_len, dev))
        # logits only where the next token is a summary token
        rows, cols, tg = [], [], []
        for b, s in enumerate(seqs):
            n = len(s) - pls[b]
            rows += [b] * n; cols += list(range(pls[b] - 1, len(s) - 1)); tg += s[pls[b]:]
        rows, cols = torch.tensor(rows, device=dev), torch.tensor(cols, device=dev)
        logits = self.lm.lm_head(h[rows, cols])
        return F.cross_entropy(logits.float(), torch.tensor(tg, dtype=torch.long, device=dev)), len(tg)

    # ------------------------------------------------------------------ B: encoder + FiD decoder
    def _encode_b(self, chunk_lists):
        """chunk_lists: per doc a list of int arrays (<= src_len each) -> memory [B, M, d], valid [B, M]."""
        lm, dev = self.lm, self.lm.tok_emb.weight.device
        flat = [c for cl in chunk_lists for c in cl]
        L = max(len(c) for c in flat)
        idx = torch.full((len(flat), L), self.pad, dtype=torch.long)
        for i, c in enumerate(flat):
            idx[i, :len(c)] = torch.as_tensor(c, dtype=torch.long)
        idx = idx.to(dev)
        valid = idx != self.pad
        keep = valid[:, None, None, :]
        cs = rope_cs(torch.arange(L, device=dev)[None], self.head_dim, self.theta, 1.0)
        x = lm.tok_emb(idx)
        for blk in lm.enc_blocks:
            def fn(x, blk=blk):
                h = self_attention(blk.attn, blk.norm1(x), cs, keep, False)
                x = x + _with_adapter(blk, "adapter_attn", h)
                return x + blk.ffn(blk.norm2(x))
            x = self._ckpt(fn, x)
        x = lm.enc_norm(x)
        mems, k = [], 0
        for cl in chunk_lists:
            mems.append(torch.cat([x[k + j, :len(c)] for j, c in enumerate(cl)], 0)); k += len(cl)
        M = max(m.size(0) for m in mems)
        mem = x.new_zeros(len(mems), M, x.size(-1))
        mvalid = torch.zeros(len(mems), M, dtype=torch.bool, device=dev)
        for b, m in enumerate(mems):
            mem[b, :m.size(0)] = m; mvalid[b, :m.size(0)] = True
        return mem, mvalid

    def _cross(self, attn, x, mem, mkeep, kv=None):
        B, T, _ = x.shape
        H, hd = attn.n_head, attn.head_dim
        q = attn.q(x).view(B, T, H, hd).transpose(1, 2)
        if kv is None:
            k, v = attn.kv(mem).split([H * hd, H * hd], dim=-1)
            kv = (k.view(B, -1, H, hd).transpose(1, 2), v.view(B, -1, H, hd).transpose(1, 2))
        y = F.scaled_dot_product_attention(q, kv[0], kv[1], attn_mask=mkeep)
        return attn.proj(y.transpose(1, 2).reshape(B, T, -1)), kv

    def _decode_b(self, idx, mem, mvalid, pos0=0, cache=None, cross_kv=None, s_max=0):
        lm = self.lm
        B, T = idx.shape
        cs = rope_cs(torch.arange(pos0, pos0 + T, device=idx.device)[None], self.head_dim, self.theta,
                     self.dec_rope_scale)
        mkeep = mvalid[:, None, None, :]
        x = lm.tok_emb(idx)
        for li, blk in enumerate(lm.dec_blocks):
            def fn(x, blk=blk, li=li):
                h = self_attention(blk.self_attn, blk.norm1(x), cs, None, cache is None or T > 1, None, cache, li, s_max)
                x = x + _with_adapter(blk, "adapter_attn", h)
                h, kv = self._cross(blk.cross_attn, blk.norm2(x), mem, mkeep,
                                    None if cross_kv is None else cross_kv[li])
                if cross_kv is not None and cross_kv[li] is None:
                    cross_kv[li] = kv
                x = x + _with_adapter(blk, "adapter_cross", h)
                return x + blk.ffn(blk.norm3(x))
            x = self._ckpt(fn, x)
        if cache is not None:
            cache.n += T
        return lm.dec_norm(x)

    def _loss_b(self, batch):
        dev = self.lm.tok_emb.weight.device
        mem, mvalid = self._encode_b([src for src, _, _ in batch])
        tg = [list(t) + ([self.eot] if done else []) for _, t, done in batch]
        T = max(map(len, tg))
        idx = torch.full((len(tg), T), self.pad, dtype=torch.long)
        y = torch.full((len(tg), T), -100, dtype=torch.long)
        for b, t in enumerate(tg):
            idx[b, :len(t)] = torch.tensor([self.bos] + t[:-1]); y[b, :len(t)] = torch.tensor(t)
        h = self._decode_b(idx.to(dev), mem, mvalid)
        y = y.to(dev)
        sel = y != -100
        logits = self.lm.lm_head(h[sel])
        return F.cross_entropy(logits.float(), y[sel]), int(sel.sum())

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, srcs, max_new=512, min_new=0, rep_penalty=1.0, no_repeat=0, temperature=0.0,
                 top_p=1.0, seed=0, throttle=None, amp=True):
        """srcs: list of source int arrays (A/C/D) or chunk lists (B). Returns lists of generated ids."""
        dev = self.lm.tok_emb.weight.device
        B = len(srcs)
        V = self.lm.tok_emb.num_embeddings
        g = torch.Generator(device=dev).manual_seed(seed) if temperature > 0 else None
        ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and dev.type == "cuda")
        with ac:
            if self.arch == "B":
                mem, mvalid = self._encode_b(srcs)
                cache, cross_kv = Cache(len(self.lm.dec_blocks)), [None] * len(self.lm.dec_blocks)
                s_max = max_new + 1
                h = self._decode_b(torch.full((B, 1), self.bos, device=dev), mem, mvalid, 0, cache, cross_kv, s_max)
                logits = self.lm.lm_head(h[:, -1]).float()
            else:
                prompts = [[self.bos] + list(s) + [self.sep] for s in srcs]
                P = max(map(len, prompts))
                lens = torch.tensor([len(p) for p in prompts], device=dev)
                lpad = P - lens
                idx = torch.full((B, P), self.pad, dtype=torch.long)
                for b, p in enumerate(prompts):
                    idx[b, P - len(p):] = torch.tensor(p)
                idx = idx.to(dev)
                ar = torch.arange(P, device=dev)
                valid = ar[None] >= lpad[:, None]                                    # [B, P]
                pos = (ar[None] - lpad[:, None]).clamp(min=0)
                s_max = P + max_new
                kvalid = torch.zeros(B, s_max, dtype=torch.bool, device=dev)
                kvalid[:, :P] = valid
                eye = torch.eye(P, dtype=torch.bool, device=dev)[None, None]
                causal = (ar[:, None] >= ar[None, :])[None, None]
                base = valid[:, None, None, :]
                masks = []
                for blk in self.lm.blocks:
                    if self.arch == "C":
                        m = base | eye
                    else:
                        m = causal & base
                        w = getattr(blk.attn, "window", None)
                        if w:
                            m = m & ((ar[:, None] - ar[None, :]) < w)[None, None]
                        m = m | eye
                    masks.append((m, False, None))
                cache = Cache(len(self.lm.blocks))
                cs = rope_cs(pos, self.head_dim, self.theta, self.rope_scale)
                h = self._stack(self._embed(idx, torch.zeros_like(idx)), cs, masks, cache, s_max)
                logits = self.lm.lm_head(h[:, -1]).float()
                cur_pos = lens.clone()                                               # next position id
            out = torch.full((B, max_new), self.pad, dtype=torch.long, device=dev)
            done = torch.zeros(B, dtype=torch.bool, device=dev)
            seen = torch.zeros(B, V, dtype=torch.bool, device=dev)
            ngrams = [dict() for _ in range(B)]
            hist = [[] for _ in range(B)]
            for t in range(max_new):
                logits[:, self.banned] = -float("inf")
                if t < min_new:
                    logits[:, self.eot] = -float("inf")
                if rep_penalty != 1.0:
                    pen = torch.where(logits > 0, logits / rep_penalty, logits * rep_penalty)
                    logits = torch.where(seen, pen, logits)
                if no_repeat and t >= no_repeat - 1:
                    for b in range(B):
                        ban = ngrams[b].get(tuple(hist[b][-(no_repeat - 1):]))
                        if ban:
                            logits[b, list(ban)] = -float("inf")
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, -1)
                    if top_p < 1.0:
                        sp, si = probs.sort(-1, descending=True)
                        cut = sp.cumsum(-1) - sp > top_p
                        sp[cut] = 0
                        probs = torch.zeros_like(probs).scatter(-1, si, sp)
                    nxt = torch.multinomial(probs, 1, generator=g).squeeze(-1)
                else:
                    nxt = logits.argmax(-1)
                nxt = torch.where(done, torch.full_like(nxt, self.pad), nxt)
                out[:, t] = nxt
                seen[torch.arange(B, device=dev), nxt] = True
                done = done | (nxt == self.eot)
                if bool(done.all()):
                    break
                nl = nxt.tolist()
                if no_repeat:
                    for b in range(B):
                        hist[b].append(nl[b])
                        if len(hist[b]) >= no_repeat:
                            ngrams[b].setdefault(tuple(hist[b][-no_repeat:-1]), set()).add(hist[b][-1])
                step = nxt[:, None]
                if self.arch == "B":
                    h = self._decode_b(step, mem, mvalid, t + 1, cache, cross_kv, s_max)
                else:
                    kvalid[:, P + t] = True
                    m = kvalid[:, None, None, :P + t + 1]
                    masks = []
                    for blk in self.lm.blocks:
                        w = getattr(blk.attn, "window", None)
                        if w:
                            far = torch.arange(P + t + 1, device=dev) <= (P + t - w)
                            masks.append((m & ~far[None, None, None], False, None))
                        else:
                            masks.append((m, False, None))
                    cs = rope_cs(cur_pos[:, None], self.head_dim, self.theta, self.rope_scale)
                    h = self._stack(self._embed(step, torch.ones_like(step)), cs, masks, cache, s_max)
                    cur_pos += 1
                logits = self.lm.lm_head(h[:, -1]).float()
                if throttle is not None and t % 16 == 15:
                    throttle.pause()
        res = []
        for row in out.tolist():
            ids = []
            for i in row:
                if i in (self.eot, self.pad):
                    break
                ids.append(i)
            res.append(ids)
        return res


def load_summarizer(name, max_len=2048, rope_scale=None, dec_rope_scale=None, device="cpu"):
    """name: key of BACKBONES. rope_scale default: max_len / pretrained block size (>= 1)."""
    run_dir = PHASE1_RUNS / BACKBONES.get(name, name)
    ck = torch.load(run_dir / "model_final.pt", map_location="cpu", weights_only=False)
    cls_name = ck["model_class"]
    lm = _CLASSES[cls_name](**ck["model_kw"])
    lm.load_state_dict(ck["model"])
    arch = _ARCH[cls_name]
    tok = BPETokenizer.load(run_dir / ck["tokenizer"])
    if arch == "B":
        rope_scale = 1.0
        dec_rope_scale = dec_rope_scale or 1.0
    else:
        rope_scale = rope_scale or max(1.0, max_len / lm.block_size)
    theta = ck["model_kw"].get("rope_theta", 10000.0)
    model = Summarizer(lm, arch, tok, max_len, rope_scale, dec_rope_scale or 1.0)
    model.theta = theta
    info = dict(name=name, run_dir=str(run_dir), arch=arch, tokenizer=ck["tokenizer"], model_kw=ck["model_kw"],
                params=sum(p.numel() for p in lm.parameters()), rope_scale=rope_scale,
                dec_rope_scale=model.dec_rope_scale)
    return model.to(device), tok, info
