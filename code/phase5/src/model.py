"""
Multi-label document classifier: Phase-1 backbone (+ adaptation) encodes each 512-token chunk of a judgment,
the chunk's token states are mean-pooled, and a head turns the chunk vectors of a document into 100 statute
logits (sigmoid / one-vs-rest). Two heads:
  att        (Phase 4 head) LayerNorm -> + chunk-position feature -> ONE attention pooling over chunks
             -> dropout -> linear -> 100 logits
  label_att  same chunk features, then chunk-level statute logits z [n, 100] and a separate attention
             distribution over chunks for every statute a [n, 100]: logit_l = sum_n a[n, l] * z[n, l]
             (statutes can be evidenced in different parts of the document)
The output bias starts at the log-odds of each statute's train prior, so BCE starts at a sensible loss.
"""
import math

import numpy as np
import torch
import torch.nn as nn


class DocHead(nn.Module):
    def __init__(self, d, n_labels=100, head="att", prior=None, dropout=0.1, n_freq=8):
        super().__init__()
        self.kind = head
        self.norm = nn.LayerNorm(d)
        self.register_buffer("freq", torch.arange(1, n_freq + 1).float() * math.pi)
        self.pos = nn.Linear(2 * n_freq, d)
        nn.init.zeros_(self.pos.weight); nn.init.zeros_(self.pos.bias)
        self.drop = nn.Dropout(dropout)
        if head == "att":
            self.att_w, self.att_v = nn.Linear(d, 256), nn.Linear(256, 1, bias=False)
        elif head == "label_att":
            self.att = nn.Linear(d, n_labels)                       # one attention query per statute
            nn.init.zeros_(self.att.weight); nn.init.zeros_(self.att.bias)   # starts as mean pooling
        else:
            raise ValueError(head)
        self.out = nn.Linear(d, n_labels)
        nn.init.normal_(self.out.weight, std=0.02)
        if prior is not None:
            p = torch.as_tensor(prior, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
            self.out.bias.data.copy_(torch.log(p / (1 - p)))
        else:
            nn.init.zeros_(self.out.bias)

    def forward(self, vecs, pos):                                # [n, d], [n] -> [100], attention weights
        a = pos[:, None] * self.freq
        x = self.drop(self.norm(vecs.float()) + self.pos(torch.cat([a.sin(), a.cos()], -1)))
        if self.kind == "att":
            w = torch.softmax(self.att_v(torch.tanh(self.att_w(x))).squeeze(-1), 0)
            return self.out(self.drop((w[:, None] * x).sum(0))), w
        w = torch.softmax(self.att(x), 0)                        # [n, 100], softmax over chunks per statute
        return (w * self.out(self.drop(x))).sum(0), w


class DocModel(nn.Module):
    def __init__(self, backbone, n_labels=100, head="att", prior=None, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.head = DocHead(backbone.d_model, n_labels, head, prior, dropout)

    def encode(self, rows, pad_id, device, token_budget, throttle=None):
        """Chunks (int arrays) -> mean-pooled vectors [n, d]; length-sorted micro-batches of <= token_budget
        padded tokens, each followed by the throttle's pause."""
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
        vecs, i = [None] * len(rows), 0
        while i < len(order):
            j = i + 1
            while j < len(order) and len(rows[order[j]]) * (j + 1 - i) <= token_budget:
                j += 1
            batch = order[i:j]
            T = len(rows[batch[-1]])
            x = np.full((len(batch), T), pad_id, dtype=np.int64)
            for r, k in enumerate(batch):
                x[r, :len(rows[k])] = rows[k]
            x = torch.from_numpy(x).to(device)
            lens = torch.tensor([len(rows[k]) for k in batch], device=device)
            key_pad = x == pad_id if self.backbone.arch in "BC" else None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                h = self.backbone(x, key_pad)
            m = (torch.arange(T, device=device)[None] < lens[:, None]).to(h.dtype)[..., None]
            v = ((h * m).sum(1) / lens[:, None]).float()
            for r, k in enumerate(batch):
                vecs[k] = v[r]
            if throttle is not None and not torch.is_grad_enabled():
                throttle.pause()
            i = j
        return torch.stack(vecs)

    def forward(self, docs, pad_id, device, token_budget, throttle=None):
        """docs: list of (chunks, positions) -> logits [n_docs, 100]."""
        rows = [c for cs, _ in docs for c in cs]
        vecs = self.encode(rows, pad_id, device, token_budget, throttle)
        out, k = [], 0
        for cs, pos in docs:
            out.append(self.head(vecs[k:k + len(cs)], torch.from_numpy(pos).to(device))[0])
            k += len(cs)
        return torch.stack(out)
