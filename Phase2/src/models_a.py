"""Architecture A - decoder-only LM (DecoderOnlyLM).

Copied verbatim from Phase1/Phase1_A_DecoderOnly_GQA_KVCache.ipynb (model cell) - GQA/KV-cache version; with n_kv_head=None it is the original A.
Do not edit the classes here except where marked "PHASE 2" - the Phase-1 checkpoints must load unchanged.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
#  ARCHITECTURE A - YOUR IMPLEMENTATION
# =============================================================================
# Building blocks (LLaMA-style): pre-norm RMSNorm, rotary position embeddings,
# SwiGLU feed-forward, fused scaled-dot-product attention, tied input/output embeddings,
# optional grouped-query attention (GQA) and a KV cache for fast generation.

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

    def forward(self, x, offset=0):                                        # x: [B, H, T, hd]
        T = x.size(-2)
        assert offset + T <= self.cos.size(0), f"position {offset + T} > RoPE table {self.cos.size(0)}"
        cos, sin = self.cos[offset:offset + T], self.sin[offset:offset + T]
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).type_as(x)


class KVCache:
    """
    Keys/values of every position processed so far, one entry per layer (generation only).
    With it, each new token is run through the model alone instead of re-reading the whole context.
    """
    def __init__(self, n_layer):
        self.layers = [{"k": None, "v": None} for _ in range(n_layer)]
        self.pos = 0                                   # number of positions already stored


class CausalSelfAttention(nn.Module):
    """
    Causal multi-head self-attention.
    Grouped-query attention: n_kv_head < n_head key/value heads, each shared by n_head // n_kv_head query heads.
    n_kv_head == n_head is standard multi-head attention (same weights/layout as before GQA was added).
    """
    def __init__(self, d_model, n_head, n_kv_head=None, dropout=0.0, bias=False):
        super().__init__()
        n_kv_head = n_kv_head or n_head
        assert d_model % n_head == 0, f"d_model ({d_model}) must be divisible by n_head ({n_head})"
        assert n_head % n_kv_head == 0, f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})"
        self.n_head, self.n_kv_head, self.head_dim = n_head, n_kv_head, d_model // n_head
        # one fused projection: [queries (n_head heads) | keys (n_kv_head) | values (n_kv_head)]
        self.qkv = nn.Linear(d_model, (n_head + 2 * n_kv_head) * self.head_dim, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, rope, layer_cache=None, pos=0):
        B, T, C = x.shape
        H, Hkv, hd = self.n_head, self.n_kv_head, self.head_dim
        q, k, v = self.qkv(x).split([H * hd, Hkv * hd, Hkv * hd], dim=-1)
        q = q.view(B, T, H, hd).transpose(1, 2)
        k = k.view(B, T, Hkv, hd).transpose(1, 2)
        v = v.view(B, T, Hkv, hd).transpose(1, 2)
        q, k = rope(q, pos), rope(k, pos)
        if layer_cache is not None:                    # append the new keys/values to the ones already cached
            assert T == 1 or pos == 0, "cache supports one prefill, then one token at a time"
            if layer_cache["k"] is not None:
                k = torch.cat([layer_cache["k"], k], dim=2)
                v = torch.cat([layer_cache["v"], v], dim=2)
            layer_cache["k"], layer_cache["v"] = k, v  # stored with n_kv_head heads (GQA keeps the cache small)
        if Hkv != H:                                   # share each K/V head across its group of query heads
            k = k.repeat_interleave(H // Hkv, dim=1)
            v = v.repeat_interleave(H // Hkv, dim=1)
        # T > 1: training or cache prefill -> causal mask. T == 1: the single new token may see every cached key.
        # PHASE 2: `bidirectional` (default False = Phase-1 behaviour) drops the causal mask for NER fine-tuning
        y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1 and not getattr(self, "bidirectional", False),
                                           dropout_p=self.dropout if self.training else 0.0)
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
    """Pre-norm residual block: x + Attn(Norm(x)); x + FFN(Norm(x))."""
    def __init__(self, d_model, n_head, n_kv_head, ffn_hidden, dropout, bias):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d_model), RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, n_kv_head, dropout, bias)
        self.ffn = SwiGLU(d_model, ffn_hidden, dropout, bias)

    def forward(self, x, rope, layer_cache=None, pos=0):
        x = x + self.attn(self.norm1(x), rope, layer_cache, pos)
        return x + self.ffn(self.norm2(x))


class DecoderOnlyLM(nn.Module):
    """
    Decoder-only Transformer language model.

    Interface the scaffold relies on
    --------------------------------
    __init__(self, vocab_size, block_size, pad_id, **CFG["model"])
    forward(idx: Long[B,T], targets: Long[B,T] | None = None) -> (logits: [B,T,V], loss | None)
        * causal self-attention: position t sees positions <= t only
        * loss = cross-entropy(logits, targets, ignore_index=-100); None if targets is None
    self.block_size : int   (generation crops context to this length)
    Generation extras: new_cache() -> KVCache; forward(idx, cache=cache) processes only the new tokens.

    Design choices
    --------------
    positions  : RoPE (relative, no learned table, extrapolates more gracefully than absolute embeddings)
    norm       : pre-norm RMSNorm + final RMSNorm (stable training without careful LR tuning)
    attention  : causal attention via F.scaled_dot_product_attention (flash / mem-efficient kernels);
                 optional grouped-query attention (n_kv_head < n_head) as in LLaMA 2/3
    FFN        : SwiGLU, hidden = ffn_mult * d_model rounded to a multiple of 64
    tying      : token embedding shared with the LM head (saves V*d params, usually better at this scale)
    init       : N(0, 0.02); residual output projections scaled by 1/sqrt(2*n_layer) (GPT-2)
    dropout    : on attention probs, after the attention/FFN output projections, and on embeddings
    bias       : no biases in linear layers (fewer params, no quality loss in modern LMs)
    generation : KV cache - each new token attends to stored keys/values instead of recomputing the context
    """
    def __init__(self, vocab_size, block_size, pad_id=None, n_layer=8, n_head=8, d_model=560,
                 dropout=0.0, ffn_mult=4.0, rope_theta=10000.0, tie_weights=True, bias=False,
                 n_kv_head=None, **kw):
        super().__init__()
        if kw:
            print(f"[DecoderOnlyLM] ignoring unknown config keys: {sorted(kw)}")
        assert d_model % n_head == 0, f"d_model ({d_model}) must be divisible by n_head ({n_head})"
        self.block_size, self.pad_id, self.n_layer = block_size, pad_id, n_layer
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(d_model // n_head, block_size, rope_theta)
        hidden = _ffn_hidden(d_model, ffn_mult)
        self.blocks = nn.ModuleList([Block(d_model, n_head, n_kv_head, hidden, dropout, bias)
                                     for _ in range(n_layer)])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):                       # attention-out and FFN-out projections
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def new_cache(self):
        return KVCache(self.n_layer)

    def forward(self, idx, targets=None, cache=None):
        B, T = idx.shape
        pos = cache.pos if cache is not None else 0
        assert pos + T <= self.block_size, f"sequence length {pos + T} > block_size {self.block_size}"
        x = self.drop(self.tok_emb(idx))
        layer_caches = cache.layers if cache is not None else [None] * self.n_layer
        for blk, layer_cache in zip(self.blocks, layer_caches):
            x = blk(x, self.rope, layer_cache, pos)
        if cache is not None:
            cache.pos += T
        logits = self.lm_head(self.norm_f(x))
        loss = None if targets is None else F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

MODEL_CLASS = DecoderOnlyLM


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
