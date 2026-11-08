"""Architecture D - efficient LM (EfficientLM: sliding-window + global layers, GQA).

Copied verbatim from Phase1/Phase1_D_EfficientTransformer.ipynb (model cell).
Do not edit the classes here except where marked "PHASE 2" - the Phase-1 checkpoints must load unchanged.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
#  ARCHITECTURE D - YOUR IMPLEMENTATION
# =============================================================================
# Efficiency ideas combined here:
#   1. Blocked sliding-window attention: each query sees only the last `window` tokens.
#      Implemented by chunking the sequence into blocks of `window` and letting every block
#      attend to itself + the previous block -> O(T * window) compute/memory instead of O(T^2).
#   2. Grouped-query attention (GQA): n_kv_head < n_head key/value heads shared across query
#      groups -> smaller K/V projections and activations.
#   3. Hybrid local/global stack: every `global_every`-th layer is full causal attention so long-range
#      information still flows (global_every=0 -> purely local, fully sub-quadratic model).
# Remaining blocks as in A: pre-norm RMSNorm, RoPE, SwiGLU, tied embeddings.

def _ffn_hidden(d_model, mult, multiple_of=64):
    h = int(mult * d_model)
    return multiple_of * ((h + multiple_of - 1) // multiple_of)


class RMSNorm(nn.Module):
    """Root-mean-square norm (no mean-centering, no bias). Computed in fp32 for AMP stability."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).type_as(x)


class RotaryEmbedding(nn.Module):
    """RoPE: rotates (q, k) pairs by a position-dependent angle -> relative-position attention, no learned table."""
    def __init__(self, head_dim, max_len, theta=10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head_dim"
        inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        freqs = torch.outer(torch.arange(max_len).float(), inv_freq)       # [max_len, hd/2]
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x):                                                  # x: [B, H, T, hd]
        T = x.size(-2)
        assert T <= self.cos.size(0), f"sequence length {T} > RoPE table {self.cos.size(0)}"
        cos, sin = self.cos[:T], self.sin[:T]
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)


def sliding_window_attention(q, k, v, window, dropout_p=0.0):
    """
    Causal attention where query t attends keys (t-window, t].  q, k, v: [B, H, T, hd].
    The sequence is cut into blocks of `window`; block c attends to [block c-1, block c] with a
    [window, 2*window] mask, so cost is O(T * window) rather than O(T^2).
    """
    B, H, T, hd = q.shape
    W = window
    if T <= W:                                   # whole sequence fits in one window -> plain causal
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout_p)
    pad = (-T) % W                               # right-pad; causality keeps padding invisible to real tokens
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    n = (T + pad) // W
    q = q.view(B, H, n, W, hd)
    k = k.view(B, H, n, W, hd)
    v = v.view(B, H, n, W, hd)
    # previous block (zeros for block 0, masked out below), concatenated with the current block
    k = torch.cat([F.pad(k, (0, 0, 0, 0, 1, 0))[:, :, :-1], k], dim=3)       # [B,H,n,2W,hd]
    v = torch.cat([F.pad(v, (0, 0, 0, 0, 1, 0))[:, :, :-1], v], dim=3)
    i = torch.arange(W, device=q.device)[:, None]                          # query offset in block
    j = torch.arange(2 * W, device=q.device)[None, :] - W                  # key offset rel. to block start
    band = (j <= i) & (j > i - W)                                          # [W, 2W]
    mask = band.expand(n, W, 2 * W).clone()
    mask[0, :, :W] = False                                                 # block 0 has no previous block
    mask = mask.repeat(H, 1, 1)[None]                                      # [1, H*n, W, 2W] (head-major)
    y = F.scaled_dot_product_attention(q.reshape(B, H * n, W, hd), k.reshape(B, H * n, 2 * W, hd),
                                       v.reshape(B, H * n, 2 * W, hd), attn_mask=mask, dropout_p=dropout_p)
    return y.view(B, H, n * W, hd)[:, :, :T]


class GQAttention(nn.Module):
    """Causal grouped-query attention; local (sliding window) or global (full causal)."""
    def __init__(self, d_model, n_head, n_kv_head, window=None, dropout=0.0, bias=False):
        super().__init__()
        assert d_model % n_head == 0 and n_head % n_kv_head == 0
        self.n_head, self.n_kv_head, self.head_dim = n_head, n_kv_head, d_model // n_head
        self.window = window                                              # None -> global attention
        self.q = nn.Linear(d_model, d_model, bias=bias)
        self.kv = nn.Linear(d_model, 2 * n_kv_head * self.head_dim, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, rope):
        B, T, C = x.shape
        H, Hkv, hd = self.n_head, self.n_kv_head, self.head_dim
        q = self.q(x).view(B, T, H, hd).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, Hkv, hd).permute(2, 0, 3, 1, 4)
        q, k = rope(q), rope(k)
        if Hkv != H:                                                      # share each K/V head across its group
            k = k.repeat_interleave(H // Hkv, dim=1)
            v = v.repeat_interleave(H // Hkv, dim=1)
        p = self.dropout if self.training else 0.0
        if getattr(self, "bidirectional", False):
            # PHASE 2: bidirectional variant for NER - global layers see everything,
            # local layers see a symmetric band |i - j| < window
            if self.window is None or T <= self.window:
                y = F.scaled_dot_product_attention(q, k, v, dropout_p=p)
            else:
                pos = torch.arange(T, device=x.device)
                band = (pos[:, None] - pos[None, :]).abs() < self.window
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=band[None, None], dropout_p=p)
        elif self.window is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
        else:
            y = sliding_window_attention(q, k, v, self.window, p)
        return self.resid_drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.0, bias=False):
        super().__init__()
        self.w12 = nn.Linear(d_model, 2 * hidden, bias=bias)
        self.proj = nn.Linear(hidden, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.drop(self.proj(F.silu(a) * b))


class Block(nn.Module):
    def __init__(self, d_model, n_head, n_kv_head, window, ffn_hidden, dropout, bias):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d_model), RMSNorm(d_model)
        self.attn = GQAttention(d_model, n_head, n_kv_head, window, dropout, bias)
        self.ffn = SwiGLU(d_model, ffn_hidden, dropout, bias)

    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        return x + self.ffn(self.norm2(x))


class EfficientLM(nn.Module):
    """
    Efficient / lightweight causal Transformer.

    It must differ *substantially* from notebook A (Sec. 7.3). Only changing width/depth/heads does NOT count.
    Families you could study: local / sliding-window attention, linear (kernelised) attention,
    sparse or chunked attention, low-rank attention, recurrent-memory hybrids, parameter sharing across layers.

    Interface the scaffold relies on (identical to A)
    -------------------------------------------------
    __init__(self, vocab_size, block_size, pad_id, **CFG["model"])
    forward(idx: Long[B,T], targets: Long[B,T] | None = None) -> (logits [B,T,V], loss | None)
        * strictly causal; loss = cross-entropy(..., ignore_index=-100)
    self.block_size : int

    Things to measure and report: tokens/s, peak GPU memory, val loss at equal tokens / equal time vs A.

    Design (differences from A)
    ---------------------------
    attention : blocked sliding-window causal attention (window `window`) in local layers,
                O(T*window) instead of O(T^2); receptive field still grows by `window` per layer
    globals   : layers where (layer+1) % global_every == 0 use full causal attention (hybrid, as in
                Mistral/Gemma-2 style models); global_every=0 -> every layer is local
    GQA       : n_kv_head key/value heads shared by n_head query heads (fewer params, smaller K/V)
    rest      : pre-norm RMSNorm, RoPE, SwiGLU, tied embeddings, GPT-2-style scaled init (same as A,
                so the comparison isolates the attention changes)
    """
    def __init__(self, vocab_size, block_size, pad_id=None, n_layer=8, n_head=8, d_model=560,
                 dropout=0.0, ffn_mult=4.0, rope_theta=10000.0, tie_weights=True, bias=False,
                 window=256, global_every=4, n_kv_head=2, **kw):
        super().__init__()
        if kw:
            print(f"[EfficientLM] ignoring unknown config keys: {sorted(kw)}")
        self.block_size, self.pad_id, self.n_layer = block_size, pad_id, n_layer
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(d_model // n_head, block_size, rope_theta)
        hidden = _ffn_hidden(d_model, ffn_mult)
        self.layer_types = ["global" if global_every and (l + 1) % global_every == 0 else "local"
                            for l in range(n_layer)]
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, n_kv_head, None if t == "global" else window, hidden, dropout, bias)
            for t in self.layer_types])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):                       # attention-out and FFN-out projections
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        print(f"[EfficientLM] layers: {self.layer_types} | window={window} | kv heads={n_kv_head}/{n_head}")

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} > block_size {self.block_size}"
        x = self.drop(self.tok_emb(idx))
        for blk in self.blocks:
            x = blk(x, self.rope)
        logits = self.lm_head(self.norm_f(x))
        loss = None if targets is None else F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

MODEL_CLASS = EfficientLM


class SmokeTestModel(nn.Module):
    """NOT an architecture: embedding -> linear bigram model, used only to test the pipeline (SMOKE_TEST)."""
    def __init__(self, vocab_size, block_size, d_model=32, **_):
        super().__init__()
        self.block_size = block_size
        self.emb, self.head = nn.Embedding(vocab_size, d_model), nn.Linear(d_model, vocab_size)
    def forward(self, idx, targets=None):
        logits = self.head(self.emb(idx))
        loss = None if targets is None else F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss
