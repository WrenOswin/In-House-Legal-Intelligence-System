"""
Fine-tuning / parameter-efficient adaptation methods, written from scratch so they work on all four
Phase-1 architectures (fused qkv / w12 projections included). Phase 2 methods + rsLoRA, IA3, prefix tuning.

method       what is trained (besides the task head)
full         every backbone weight
partial      the top `n_train_layers` blocks + final norm (layer-wise / partial fine-tuning)
lora         W x + (alpha/r) * B A x                         (Hu et al. 2021)          B=0 init
rslora       W x + (alpha/sqrt(r)) * B A x                   (Kalajdzievski 2023)      rank-stabilised scaling
lorma        W x + (alpha/r) * B A (W x)  = (I + a/r BA) W x  (LoRMA+, Bihany et al. 2025) B=0 init
ia3          keys, values and the FFN inner activation rescaled by learned vectors (Liu et al. 2022), init 1
prefix       n_prefix learned key/value vectors prepended in every attention layer (Li & Liang 2021),
             produced by a 2-layer MLP from a prefix embedding (the reparameterisation of the paper)
adapter      Pfeiffer bottleneck adapter after each FFN output
PHASE 4:
bitfit       bias terms only (Ben Zaken et al. 2022). The Phase-1 backbones have no biases, so a zero-init
             bias vector is added to every linear layer of every block and only those vectors are trained
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# linear layers inside transformer blocks, by attribute name (A/C: qkv; B: qkv; D: q, kv; all: proj, w12)
DEFAULT_TARGETS = ("qkv", "q", "kv", "proj", "w12")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0, rank_stabilised=False):
        super().__init__()
        self.base, self.r = base, r
        self.scale = alpha / math.sqrt(r) if rank_stabilised else alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.scale * F.linear(F.linear(self.drop(x), self.A), self.B)


class LoRMALinear(nn.Module):
    """LoRMA+ (additive rank inflation): h = (I + alpha/r * B A) W0 x, computed as W0x + s * B(A(W0x))
    so the cost is O(d r) per token like LoRA. B = 0 at init -> starts exactly at the pretrained model."""
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        d = base.out_features
        self.A = nn.Parameter(torch.empty(r, d))
        self.B = nn.Parameter(torch.zeros(d, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.base(x)
        return h + self.scale * F.linear(F.linear(self.drop(h), self.A), self.B)


class IA3Out(nn.Module):
    """Rescales output features [skip:] of a linear layer (skip = the query part of a fused qkv)."""
    def __init__(self, base: nn.Linear, skip=0):
        super().__init__()
        self.base, self.skip = base, skip
        self.l = nn.Parameter(torch.ones(base.out_features - skip))

    def forward(self, x):
        h = self.base(x)
        if self.skip == 0:
            return h * self.l
        return torch.cat([h[..., :self.skip], h[..., self.skip:] * self.l], -1)


class IA3In(nn.Module):
    """Rescales the input features of a linear layer (the FFN inner activation before the down-projection)."""
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.l = nn.Parameter(torch.ones(base.in_features))

    def forward(self, x):
        return self.base(x * self.l)


class Adapter(nn.Module):
    """Bottleneck adapter applied to a sub-layer's output: out + up(act(down(out))); up = 0 at init."""
    def __init__(self, sublayer, d_model, bottleneck=64):
        super().__init__()
        self.sublayer = sublayer
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)

    def forward(self, *a, **kw):
        out = self.sublayer(*a, **kw)
        return out + self.up(F.gelu(self.down(out)))


class PrefixEncoder(nn.Module):
    """[n_prefix, d] embedding -> MLP -> per-layer key and value prefixes [n_layer, 2, H, n_prefix, hd]."""
    def __init__(self, n_layer, n_head, head_dim, d_model, n_prefix=16, hidden=512):
        super().__init__()
        self.shape = (n_prefix, n_layer, 2, n_head, head_dim)
        self.emb = nn.Parameter(torch.randn(n_prefix, d_model) * 0.02)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.Tanh(),
                                 nn.Linear(hidden, n_layer * 2 * n_head * head_dim))

    def forward(self):
        return self.mlp(self.emb).view(*self.shape).permute(1, 2, 3, 0, 4)


def _sliding_window_prefix(q, k, v, pk, pv, W, dropout_p=0.0):
    """D's blocked causal sliding-window attention (models_d.sliding_window_attention) with the prefix
    keys/values added to every block: memory O(T * (2W + P)) instead of a dense [T, T + P] mask."""
    B, H, T, hd = q.shape
    P = pk.size(1)
    pad = (-T) % W
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    n = (T + pad) // W
    q, k, v = (t.view(B, H, n, W, hd) for t in (q, k, v))
    pre_k = pk.to(k.dtype)[None, :, None].expand(B, H, n, P, hd)
    pre_v = pv.to(v.dtype)[None, :, None].expand(B, H, n, P, hd)
    k = torch.cat([pre_k, F.pad(k, (0, 0, 0, 0, 1, 0))[:, :, :-1], k], dim=3)        # [B,H,n,P+2W,hd]
    v = torch.cat([pre_v, F.pad(v, (0, 0, 0, 0, 1, 0))[:, :, :-1], v], dim=3)
    i = torch.arange(W, device=q.device)[:, None]
    j = torch.arange(2 * W, device=q.device)[None, :] - W
    band = (j <= i) & (j > i - W)
    mask = band.expand(n, W, 2 * W).clone()
    mask[0, :, :W] = False
    mask = torch.cat([torch.ones(n, W, P, dtype=torch.bool, device=q.device), mask], -1)
    mask = mask.repeat(H, 1, 1)[None]                                                # [1, H*n, W, P+2W]
    y = F.scaled_dot_product_attention(q.reshape(B, H * n, W, hd), k.reshape(B, H * n, P + 2 * W, hd),
                                       v.reshape(B, H * n, P + 2 * W, hd), attn_mask=mask, dropout_p=dropout_p)
    return y.view(B, H, n * W, hd)[:, :, :T]


def prefix_attention(q, k, v, pk, pv, causal=False, window=None, base_mask=None, dropout_p=0.0):
    """Attention over [prefix ; sequence] keys. q, k, v: [B, H, T, hd] (k/v already expanded to H heads);
    pk, pv: [H, P, hd]. The sequence part keeps its usual mask (base_mask, or causal / sliding window);
    the prefix is visible to every query."""
    B, H, T, hd = q.shape
    P = pk.size(1)
    if base_mask is None and causal and window and T > window:
        return _sliding_window_prefix(q, k, v, pk, pv, window, dropout_p)
    if base_mask is None:
        i = torch.arange(T, device=q.device)[:, None]
        j = torch.arange(T, device=q.device)[None, :]
        m = j <= i if causal else torch.ones(T, T, dtype=torch.bool, device=q.device)
        if window:
            m = m & ((i - j) < window) if causal else m & ((i - j).abs() < window)
        m = m[None, None]
    else:
        m = base_mask.bool().expand(base_mask.size(0), base_mask.size(1), T, base_mask.size(-1))
    m = torch.cat([torch.ones(*m.shape[:-1], P, dtype=torch.bool, device=q.device), m], -1)
    k = torch.cat([pk.to(k.dtype)[None].expand(B, -1, -1, -1), k], 2)
    v = torch.cat([pv.to(v.dtype)[None].expand(B, -1, -1, -1), v], 2)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=m, dropout_p=dropout_p)


def _replace_linears(blocks, make, targets):
    n = 0
    for blk in blocks:
        for parent in blk.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear) and name in targets:
                    setattr(parent, name, make(child)); n += 1
    return n


def attach_prefix(backbone, n_prefix=16, hidden=512):
    blocks = backbone.blocks
    a0 = blocks[0].attn
    enc = PrefixEncoder(len(blocks), a0.n_head, a0.head_dim, backbone.d_model, n_prefix, hidden)
    backbone.prefix_enc = enc                       # registered -> trained and saved with the model
    for li, blk in enumerate(blocks):
        def fn(li=li):
            out = enc()
            return out[li, 0], out[li, 1]
        blk.attn.prefix_fn = fn                     # plain function attribute, not a submodule
    return enc


def apply_method(backbone, method, r=8, alpha=16, dropout=0.0, n_train_layers=4,
                 bottleneck=64, targets=DEFAULT_TARGETS, n_prefix=16, prefix_hidden=512):
    """Freezes/wraps the backbone in place. Returns a short description string."""
    blocks = backbone.blocks
    if method == "full":
        for p in backbone.parameters():
            p.requires_grad = True
        return "full fine-tuning"
    for p in backbone.parameters():
        p.requires_grad = False
    if method == "bitfit":
        n = 0
        for blk in blocks:
            for m in blk.modules():
                if isinstance(m, nn.Linear):
                    if m.bias is None:
                        m.bias = nn.Parameter(torch.zeros(m.out_features, device=m.weight.device,
                                                          dtype=m.weight.dtype))
                    m.bias.requires_grad = True
                    n += 1
        return f"bitfit: zero-init biases added to {n} linear layers, only biases trained"
    if method == "partial":
        for blk in blocks[-n_train_layers:]:
            for p in blk.parameters():
                p.requires_grad = True
        norm = backbone.lm.enc_norm if backbone.arch == "B" else backbone.lm.norm_f
        for p in norm.parameters():
            p.requires_grad = True
        return f"partial: top {n_train_layers}/{len(blocks)} blocks + final norm"
    if method in ("lora", "rslora", "lorma"):
        if method == "lorma":
            make = lambda m: LoRMALinear(m, r=r, alpha=alpha, dropout=dropout)
        else:
            make = lambda m: LoRALinear(m, r=r, alpha=alpha, dropout=dropout, rank_stabilised=method == "rslora")
        n = _replace_linears(blocks, make, targets)
        return f"{method}: r={r} alpha={alpha} on {n} linear layers ({','.join(targets)})"
    if method == "ia3":
        for blk in blocks:
            attn = blk.attn
            if hasattr(attn, "qkv"):
                attn.qkv = IA3Out(attn.qkv, skip=attn.n_head * attn.head_dim)
            else:                                   # D: separate q and kv projections
                attn.kv = IA3Out(attn.kv)
            blk.ffn.proj = IA3In(blk.ffn.proj)
        return f"ia3: k/v + FFN rescaling vectors in {len(blocks)} blocks"
    if method == "prefix":
        attach_prefix(backbone, n_prefix, prefix_hidden)
        return f"prefix tuning: {n_prefix} prefix vectors per layer (MLP hidden {prefix_hidden})"
    if method == "adapter":
        d = backbone.d_model
        for blk in blocks:
            blk.ffn = Adapter(blk.ffn, d, bottleneck)
        return f"pfeiffer adapters, bottleneck {bottleneck}"
    raise ValueError(method)


def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, train
