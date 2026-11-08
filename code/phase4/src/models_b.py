"""Architecture B - encoder-decoder LM (EncoderDecoderLM). Phase 2 uses only encode().

Copied verbatim from Phase1/Phase1_B_EncoderDecoder.ipynb (model cell).
Do not edit the classes here except where marked "PHASE 2" or "PHASE 3" - the Phase-1 checkpoints must load unchanged.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
#  ARCHITECTURE B - YOUR IMPLEMENTATION
# =============================================================================
# Building blocks: pre-norm RMSNorm, rotary position embeddings (self-attention only),
# SwiGLU feed-forward, fused scaled-dot-product attention, one embedding shared by
# encoder input, decoder input and LM head.

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


class Attention(nn.Module):
    """Multi-head attention. Self-attention (q=k=v source, RoPE) or cross-attention (k, v from `memory`, no RoPE)."""
    def __init__(self, d_model, n_head, dropout=0.0, bias=False, cross=False):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.head_dim, self.cross = n_head, d_model // n_head, cross
        if cross:
            self.q = nn.Linear(d_model, d_model, bias=bias)
            self.kv = nn.Linear(d_model, 2 * d_model, bias=bias)
        else:
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x, rope=None, memory=None, attn_mask=None, is_causal=False):
        B, T, C = x.shape
        H, hd = self.n_head, self.head_dim
        if self.cross:
            S = memory.size(1)
            q = self.q(x).view(B, T, H, hd).transpose(1, 2)
            k, v = self.kv(memory).view(B, S, 2, H, hd).permute(2, 0, 3, 1, 4)
        else:
            q, k, v = self.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
            q, k = rope(q), rope(k)
            # PHASE 3: prefix tuning - learned key/value prefixes (peft.attach_prefix sets `prefix_fn`)
            if getattr(self, "prefix_fn", None) is not None:
                from peft import prefix_attention
                pk, pv = self.prefix_fn()
                y = prefix_attention(q, k, v, pk, pv, causal=is_causal, base_mask=attn_mask,
                                     dropout_p=self.dropout if self.training else 0.0)
                return self.resid_drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal,
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


class EncoderBlock(nn.Module):
    """Pre-norm: x + BiAttn(Norm(x)); x + FFN(Norm(x))."""
    def __init__(self, d_model, n_head, ffn_hidden, dropout, bias):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(d_model), RMSNorm(d_model)
        self.attn = Attention(d_model, n_head, dropout, bias)
        self.ffn = SwiGLU(d_model, ffn_hidden, dropout, bias)

    def forward(self, x, rope, key_mask=None):
        x = x + self.attn(self.norm1(x), rope, attn_mask=key_mask)
        return x + self.ffn(self.norm2(x))


class DecoderBlock(nn.Module):
    """Pre-norm: x + CausalAttn(Norm(x)); x + CrossAttn(Norm(x), memory); x + FFN(Norm(x))."""
    def __init__(self, d_model, n_head, ffn_hidden, dropout, bias):
        super().__init__()
        self.norm1, self.norm2, self.norm3 = RMSNorm(d_model), RMSNorm(d_model), RMSNorm(d_model)
        self.self_attn = Attention(d_model, n_head, dropout, bias)
        self.cross_attn = Attention(d_model, n_head, dropout, bias, cross=True)
        self.ffn = SwiGLU(d_model, ffn_hidden, dropout, bias)

    def forward(self, x, memory, rope, key_mask=None):
        x = x + self.self_attn(self.norm1(x), rope, is_causal=True)
        x = x + self.cross_attn(self.norm2(x), memory=memory, attn_mask=key_mask)
        return x + self.ffn(self.norm3(x))


class EncoderDecoderLM(nn.Module):
    """
    Encoder-decoder Transformer for prompt -> continuation.

    Interface the scaffold relies on
    --------------------------------
    __init__(self, vocab_size, src_len, tgt_len, pad_id, **CFG["model"])
    encode(src: Long[B,S], src_pad_mask: Bool[B,S] | None = None) -> memory   (any tensor/tuple)
    decode(tgt_in: Long[B,T], memory, src_pad_mask=None) -> logits [B,T,V]    (causal over tgt_in)
    forward(src, tgt_in, targets=None, src_pad_mask=None) -> (logits, loss | None)
        * loss = cross-entropy(logits, targets, ignore_index=-100)
    self.src_len, self.tgt_len : int
    (src_pad_mask: True = padding. The scaffold never pads, but keep the argument.)

    Design choices
    --------------
    depth split   : n_enc_layer / n_dec_layer configurable (default 6/6, T5-style symmetric)
    cross-attn    : in every decoder block, between self-attention and FFN (standard Vaswani placement)
    embeddings    : one table shared by encoder, decoder and LM head (same BPE vocabulary on both sides)
    positions     : RoPE in encoder self-attn and decoder self-attn, each side counting from 0;
                    no positions in cross-attention (source order is already encoded in `memory`)
    norm / FFN    : pre-norm RMSNorm, final RMSNorm on both stacks, SwiGLU FFN
    init          : N(0, 0.02); residual output projections scaled by 1/sqrt(2*(n_enc+n_dec))
    objective     : prefix -> continuation teacher forcing (loss only on target tokens), set by the batching cell
    """
    def __init__(self, vocab_size, src_len, tgt_len, pad_id=None, n_enc_layer=6, n_dec_layer=6,
                 n_head=8, d_model=512, dropout=0.0, ffn_mult=4.0, rope_theta=10000.0,
                 tie_weights=True, bias=False, **kw):
        super().__init__()
        if kw:
            print(f"[EncoderDecoderLM] ignoring unknown config keys: {sorted(kw)}")
        self.src_len, self.tgt_len, self.pad_id = src_len, tgt_len, pad_id
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(d_model // n_head, max(src_len, tgt_len), rope_theta)
        hidden = _ffn_hidden(d_model, ffn_mult)
        self.enc_blocks = nn.ModuleList([EncoderBlock(d_model, n_head, hidden, dropout, bias)
                                         for _ in range(n_enc_layer)])
        self.dec_blocks = nn.ModuleList([DecoderBlock(d_model, n_head, hidden, dropout, bias)
                                         for _ in range(n_dec_layer)])
        self.enc_norm, self.dec_norm = RMSNorm(d_model), RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)
        n_total = n_enc_layer + n_dec_layer
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):                       # attention-out and FFN-out projections
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_total))
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
    def _key_mask(src_pad_mask):
        # SDPA bool mask: True = may attend. [B,S] padding mask -> [B,1,1,S] keep mask
        return None if src_pad_mask is None else (~src_pad_mask.bool())[:, None, None, :]

    def encode(self, src, src_pad_mask=None):
        assert src.size(1) <= self.src_len, f"source length {src.size(1)} > src_len {self.src_len}"
        key_mask = self._key_mask(src_pad_mask)
        x = self.drop(self.tok_emb(src))
        for blk in self.enc_blocks:
            x = blk(x, self.rope, key_mask)
        return self.enc_norm(x)

    def decode(self, tgt_in, memory, src_pad_mask=None):
        assert tgt_in.size(1) <= self.tgt_len, f"target length {tgt_in.size(1)} > tgt_len {self.tgt_len}"
        key_mask = self._key_mask(src_pad_mask)
        x = self.drop(self.tok_emb(tgt_in))
        for blk in self.dec_blocks:
            x = blk(x, memory, self.rope, key_mask)
        return self.lm_head(self.dec_norm(x))

    def forward(self, src, tgt_in, targets=None, src_pad_mask=None):
        logits = self.decode(tgt_in, self.encode(src, src_pad_mask), src_pad_mask)
        loss = None if targets is None else F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss

MODEL_CLASS = EncoderDecoderLM


class SmokeTestModel(nn.Module):
    """NOT an architecture: bag-of-embeddings 'encoder' + bigram 'decoder', only for SMOKE_TEST."""
    def __init__(self, vocab_size, src_len, tgt_len, d_model=32, **_):
        super().__init__()
        self.src_len, self.tgt_len = src_len, tgt_len
        self.emb, self.head = nn.Embedding(vocab_size, d_model), nn.Linear(d_model, vocab_size)
    def encode(self, src, src_pad_mask=None):
        return self.emb(src).mean(1, keepdim=True)
    def decode(self, tgt_in, memory, src_pad_mask=None):
        return self.head(self.emb(tgt_in) + memory)
    def forward(self, src, tgt_in, targets=None, src_pad_mask=None):
        logits = self.decode(tgt_in, self.encode(src))
        loss = None if targets is None else F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        return logits, loss
