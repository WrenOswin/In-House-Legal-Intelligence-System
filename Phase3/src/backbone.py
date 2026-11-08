"""
Load a Phase-1 model_final.pt and expose it as a token encoder: idx [B,T] -> hidden [B,T,d].

Arch  class            what Phases 2-3 use
A     DecoderOnlyLM    all blocks + final norm; causal (as pretrained) or bidirectional
B     EncoderDecoderLM encoder only (bidirectional by design, max 768 tokens); decoder is dropped
C     PrefixLM         whole window as prefix -> bidirectional (the mode C was pretrained with on prompts)
D     EfficientLM      all blocks; causal sliding-window (as pretrained) or bidirectional band/global
The LM head is never used (and dropped), so parameter counts below are encoder-only.
"""
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import models_a, models_b, models_c, models_d
from bpe import BPETokenizer

PHASE1_RUNS = Path("C:/kaggle/working/cs781/runs")

# short names used in configs / run ids
BACKBONES = {
    "A_d768": "A_decoder_only/d768_L12_T1024_v35k_5fbdf6",
    "A_gqa_d576": "A_decoder_only_gqa/d576_L23_T1024_v35k_fbb29e",
    "A_d640": "A_decoder_only/d640_L8_T1024_v35k_42d59e",
    "A_d560": "A_decoder_only/d560_L8_T1024_v32k_6613f2",
    "B_d512": "B_encoder_decoder/d512_E6D6_S768T256_v32k_05421b",
    "C_d560": "C_prefix_lm/d560_L8_T1024_v32k_6cce41",
    "D_d560": "D_efficient/d560_L8_T2048_v32k_05e1ef",
}

_CLASSES = {"DecoderOnlyLM": models_a.DecoderOnlyLM, "EncoderDecoderLM": models_b.EncoderDecoderLM,
            "PrefixLM": models_c.PrefixLM, "EfficientLM": models_d.EfficientLM}


class Backbone(nn.Module):
    def __init__(self, lm, arch, bidirectional=True):
        super().__init__()
        self.lm, self.arch = lm, arch
        if arch == "B":
            del lm.dec_blocks, lm.dec_norm                 # decoder not needed for token classification
            self.max_len = lm.src_len
        else:
            self.max_len = lm.block_size
        lm.lm_head = None                                  # tied to tok_emb; dropping it frees nothing but avoids misuse
        self.d_model = lm.tok_emb.embedding_dim
        self.grad_ckpt = False                             # PHASE 3: activation checkpointing per block
        self.set_bidirectional(bidirectional)

    def set_bidirectional(self, flag):
        """A and D only: switch the causal mask off/on (B and C are always bidirectional here)."""
        self.bidirectional = bool(flag) if self.arch in "AD" else True
        for m in self.lm.modules():
            if isinstance(m, (models_a.CausalSelfAttention, models_d.GQAttention)):
                m.bidirectional = self.bidirectional

    @property
    def blocks(self):
        return self.lm.enc_blocks if self.arch == "B" else self.lm.blocks

    def _run(self, blk, *args):
        if self.grad_ckpt and self.training and torch.is_grad_enabled():
            return checkpoint(blk, *args, use_reentrant=False)
        return blk(*args)

    def forward(self, idx, key_pad=None):
        """idx [B,T]; key_pad [B,T] bool, True = padding (right-padded batches). Padding only matters for the
        bidirectional B/C encoders; causal A/D never let a real token see the right padding."""
        lm = self.lm
        B, T = idx.shape
        keep = None if key_pad is None else (~key_pad.bool())[:, None, None, :]   # [B,1,1,T], True = may attend
        if self.arch == "B":
            x = lm.drop(lm.tok_emb(idx))
            for blk in lm.enc_blocks:
                x = self._run(blk, x, lm.rope, keep)
            return lm.enc_norm(x)
        x = lm.tok_emb(idx)
        if self.arch == "C":
            if lm.seg_emb is not None:                     # segment 0 = prefix for every position
                x = x + lm.seg_emb.weight[0]
            mask = keep if keep is not None else torch.ones(1, 1, 1, T, dtype=torch.bool, device=idx.device)
            x = lm.drop(x)
            for blk in lm.blocks:
                x = self._run(blk, x, lm.rope, mask)
        else:                                              # A, D
            x = lm.drop(x)
            for blk in lm.blocks:
                x = self._run(blk, x, lm.rope)
        return lm.norm_f(x)


def load_backbone(name, bidirectional=True, device="cpu"):
    """name: key of BACKBONES (or a run dir). Returns (Backbone, tokenizer, info dict)."""
    run_dir = PHASE1_RUNS / BACKBONES.get(name, name)
    ck = torch.load(run_dir / "model_final.pt", map_location="cpu", weights_only=False)
    cls_name = ck["model_class"]
    lm = _CLASSES[cls_name](**ck["model_kw"])
    lm.load_state_dict(ck["model"])
    arch = {"DecoderOnlyLM": "A", "EncoderDecoderLM": "B", "PrefixLM": "C", "EfficientLM": "D"}[cls_name]
    bb = Backbone(lm, arch, bidirectional).to(device)
    tok = BPETokenizer.load(run_dir / ck["tokenizer"])
    info = dict(name=name, run_dir=str(run_dir), arch=arch, model_class=cls_name, tokenizer=ck["tokenizer"],
                model_kw=ck["model_kw"], max_len=bb.max_len, d_model=bb.d_model,
                params=sum(p.numel() for p in bb.parameters()))
    return bb, tok, info
