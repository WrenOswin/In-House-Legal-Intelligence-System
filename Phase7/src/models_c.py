"""Architecture C - prefix-LM (PrefixLM). Phase 2 treats the whole window as the prefix (bidirectional).

Copied verbatim from Phase1/Phase1_C_PrefixLM_Hybrid.ipynb (model cell).
Do not edit the classes here except where marked "PHASE 2" or "PHASE 3" - the Phase-1 checkpoints must load unchanged.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
#  ARCHITECTURE C - YOUR IMPLEMENTATION
# =============================================================================
# Building blocks: pre-norm RMSNorm, rotary position embeddings, SwiGLU feed-forward,
# fused scaled-dot-product attention with a per-sample prefix-LM mask, tied embeddings,
# plus a learned prefix/suffix segment embedding.

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


class PrefixSelfAttention(nn.Module):
    """Multi-head self-attention that takes an explicit keep-mask (True = may attend) or falls back to causal."""
    def __init__(self, d_model, n_head, dropout=0.0, bias=False):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.head_dim = n_head, d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, rope, attn_mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = rope(q), rope(k)
        # PHASE 3: prefix tuning - learned key/value prefixes (peft.attach_prefix sets `prefix_fn`)
        if getattr(self, "prefix_fn", None) is not None:
            from peft import prefix_attention
            pk, pv = self.prefix_fn()
            y = prefix_attention(q, k, v, pk, pv, causal=attn_mask is None, base_mask=attn_mask,
                                 dropout_p=self.dropout if self.training else 0.0)
            return self.resid_drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=attn_mask is None,
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
    """Pre-norm residual block: x + Attn(Norm(x), mask); x + FFN(Norm(x))."""
    def __init__(self, d_model, n_head, ffn_hidden, dropout, bias):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d_model), RMSNorm(d_model)
        self.attn = PrefixSelfAttention(d_model, n_head, dropout, bias)
        self.ffn = SwiGLU(d_model, ffn_hidden, dropout, bias)

    def forward(self, x, rope, attn_mask):
        x = x + self.attn(self.norm1(x), rope, attn_mask)
        return x + self.ffn(self.norm2(x))


class PrefixLM(nn.Module):
    """
    Prefix-LM / hybrid-mask Transformer.

    Interface the scaffold relies on
    --------------------------------
    __init__(self, vocab_size, block_size, pad_id, **CFG["model"])
    forward(idx: Long[B,T], prefix_len: Long[B], targets: Long[B,T] | None = None) -> (logits [B,T,V], loss | None)
        * attention rule for sample b: query i may attend key j  iff  j < prefix_len[b]  or  j <= i
        * prefix_len = 0 -> ordinary causal LM
        * loss = cross-entropy(logits, targets, ignore_index=-100)
    self.block_size : int

    Design choices
    --------------
    mask          : built once per forward as a bool [B,1,T,T] tensor from two broadcast comparisons
                    ((j < prefix_len[b]) | (j <= i)) and shared by all layers; when every prefix_len
                    is 0 the fused is_causal kernel is used instead (no mask tensor at all)
    parameters    : one shared stack for prefix and suffix (the defining property of a prefix-LM)
    segment emb   : optional learned 2-row embedding (0 = prefix, 1 = suffix) added to token embeddings,
                    so each token knows which side of the boundary it is on (`segment_emb` knob)
    positions     : RoPE (relative; continues seamlessly across the prefix/suffix boundary)
    norm / FFN    : pre-norm RMSNorm + final RMSNorm, SwiGLU
    init / tying  : N(0, 0.02), residual projections scaled by 1/sqrt(2*n_layer); embedding tied to LM head
    """
    def __init__(self, vocab_size, block_size, pad_id=None, n_layer=8, n_head=8, d_model=560,
                 dropout=0.0, ffn_mult=4.0, rope_theta=10000.0, tie_weights=True, bias=False,
                 segment_emb=True, **kw):
        super().__init__()
        if kw:
            print(f"[PrefixLM] ignoring unknown config keys: {sorted(kw)}")
        self.block_size, self.pad_id, self.n_layer = block_size, pad_id, n_layer
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.seg_emb = nn.Embedding(2, d_model) if segment_emb else None
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(d_model // n_head, block_size, rope_theta)
        hidden = _ffn_hidden(d_model, ffn_mult)
        self.blocks = nn.ModuleList([Block(d_model, n_head, hidden, dropout, bias) for _ in range(n_layer)])
        self.norm_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):                       # attention-out and FFN-out projections
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))
        if self.seg_emb is not None:
            nn.init.zeros_(self.seg_emb.weight)                    # starts as a no-op, learned from there
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

    @staticmethod
    def prefix_mask(prefix_len, T):
        """Bool [B,1,T,T], True = query i may attend key j: (j < prefix_len[b]) or (j <= i)."""
        pos = torch.arange(T, device=prefix_len.device)
        in_prefix = pos[None, None, :] < prefix_len[:, None, None]      # [B,1,T]  (key j in prefix)
        causal = pos[None, :, None] >= pos[None, None, :]               # [1,T,T]  (j <= i)
        return (in_prefix | causal)[:, None]

    def forward(self, idx, prefix_len=None, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"sequence length {T} > block_size {self.block_size}"
        if prefix_len is None:
            prefix_len = torch.zeros(B, dtype=torch.long, device=idx.device)
        prefix_len = prefix_len.to(idx.device).long().clamp(0, T)
        attn_mask = None if bool((prefix_len == 0).all()) else self.prefix_mask(prefix_len, T)

        x = self.tok_emb(idx)
        if self.seg_emb is not None:
            seg = (torch.arange(T, device=idx.device)[None, :] >= prefix_len[:, None]).long()
            x = x + self.seg_emb(seg)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, self.rope, attn_mask)
        logits = self.lm_head(self.norm_f(x))
        loss = None if targets is None else F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

MODEL_CLASS = PrefixLM


class SmokeTestModel(nn.Module):
    """NOT an architecture: bigram model that ignores prefix_len, only for SMOKE_TEST."""
    def __init__(self, vocab_size, block_size, d_model=32, **_):
        super().__init__()
        self.block_size = block_size
        self.emb, self.head = nn.Embedding(vocab_size, d_model), nn.Linear(d_model, vocab_size)
    def forward(self, idx, prefix_len=None, targets=None):
        logits = self.head(self.emb(idx))
        loss = None if targets is None else F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss
