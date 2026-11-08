"""
Fine-tuning / parameter-efficient adaptation methods, written from scratch so they work on all four
Phase-1 architectures (fused qkv / w12 projections included).

method       what is trained (besides the NER head)
full         every backbone weight
partial      the top `n_train_layers` blocks + final norm (layer-wise / partial fine-tuning)
lora         W x + (alpha/r) * B A x                         (Hu et al. 2021)       B=0 init
lorma        W x + (alpha/r) * B A (W x)  = (I + a/r BA) W x  (LoRMA+, Bihany et al. 2025) B=0 init
adapter      Pfeiffer bottleneck adapter after each FFN output
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# linear layers inside transformer blocks, by attribute name (A/C: qkv; B: qkv (+q/kv cross, unused);
# D: q, kv; all: proj, w12)
DEFAULT_TARGETS = ("qkv", "q", "kv", "proj", "w12")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
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


def _replace_linears(blocks, make, targets):
    n = 0
    for blk in blocks:
        for parent in blk.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear) and name in targets:
                    setattr(parent, name, make(child)); n += 1
    return n


def apply_method(backbone, method, r=8, alpha=16, dropout=0.0, n_train_layers=4,
                 bottleneck=64, targets=DEFAULT_TARGETS):
    """Freezes/wraps the backbone in place. Returns a short description string."""
    blocks = backbone.blocks
    if method == "full":
        for p in backbone.parameters():
            p.requires_grad = True
        return "full fine-tuning"
    for p in backbone.parameters():
        p.requires_grad = False
    if method == "partial":
        for blk in blocks[-n_train_layers:]:
            for p in blk.parameters():
                p.requires_grad = True
        norm = backbone.lm.enc_norm if backbone.arch == "B" else backbone.lm.norm_f
        for p in norm.parameters():
            p.requires_grad = True
        return f"partial: top {n_train_layers}/{len(blocks)} blocks + final norm"
    if method in ("lora", "lorma"):
        cls = {"lora": LoRALinear, "lorma": LoRMALinear}[method]
        n = _replace_linears(blocks, lambda m: cls(m, r=r, alpha=alpha, dropout=dropout), targets)
        return f"{method}: r={r} alpha={alpha} on {n} linear layers ({','.join(targets)})"
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
