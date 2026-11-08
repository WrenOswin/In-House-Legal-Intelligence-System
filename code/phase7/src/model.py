"""
Retrieval encoder: Phase-1 backbone (+ adaptation) encodes each 512-token chunk of a case, the chunk's token
states are mean-pooled and projected (linear, identity init) to a unit-length chunk vector. A case is the set of
its chunk vectors (<= eval_chunks evenly spaced at inference). Query-candidate scores:
  doc     cosine of the length-normalised means of the chunk vectors (one vector per case)
  maxsim  mean over query chunks of the best-matching candidate chunk (late interaction at chunk level,
          ColBERT-style but over 512-token passages: a query that quotes / follows one part of a prior case
          only needs that part to match)
The same function scores training batches (sampled chunks) and full pools (all chunks).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkEncoder(nn.Module):
    def __init__(self, backbone, dim=None):
        super().__init__()
        self.backbone = backbone
        d = backbone.d_model
        self.proj = nn.Linear(d, dim or d, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(dim or d, d))

    def forward(self, rows, pad_id, device, token_budget, throttle=None):
        """Chunks (int arrays) -> unit chunk vectors [n, dim]; length-sorted micro-batches of <= token_budget
        padded tokens (Phase-5 DocModel.encode), each followed by the throttle's pause at inference."""
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
            v = F.normalize(self.proj(((h * m).sum(1) / lens[:, None]).float()), dim=-1)
            for r, k in enumerate(batch):
                vecs[k] = v[r]
            if throttle is not None and not torch.is_grad_enabled():
                throttle.pause()
            i = j
        return torch.stack(vecs)


def pad_sets(vecs, owner, n_docs):
    """Flat chunk vectors [N, d] + owner doc index [N] -> padded [n_docs, C, d] and mask [n_docs, C]."""
    owner = torch.as_tensor(owner, device=vecs.device)
    counts = torch.bincount(owner, minlength=n_docs)
    C = int(counts.max())
    slot = torch.zeros_like(owner)
    order = torch.argsort(owner, stable=True)
    starts = torch.cumsum(counts, 0) - counts
    slot[order] = torch.arange(len(owner), device=vecs.device) - starts[owner[order]]
    out = vecs.new_zeros(n_docs, C, vecs.size(1))
    mask = torch.zeros(n_docs, C, dtype=torch.bool, device=vecs.device)
    out[owner, slot] = vecs
    mask[owner, slot] = True
    return out, mask


def doc_vectors(sets, mask):
    return F.normalize((sets * mask[..., None]).sum(1), dim=-1)


def score(q_sets, q_mask, c_sets, c_mask, kind="maxsim"):
    """[nq, Cq, d], [nc, Cc, d] -> [nq, nc] similarity in [-1, 1]."""
    if kind == "doc":
        return doc_vectors(q_sets, q_mask) @ doc_vectors(c_sets, c_mask).T
    sim = torch.einsum("qid,cjd->qcij", q_sets, c_sets)                     # [nq, nc, Cq, Cc]
    sim = sim.masked_fill(~c_mask[None, :, None, :], -2.0).amax(-1)         # best candidate chunk per query chunk
    w = q_mask[:, None, :].to(sim.dtype)
    return (sim * w).sum(-1) / w.sum(-1)


def score_blocks(q_sets, q_mask, c_sets, c_mask, kind="maxsim", block=64):
    """score() over a whole pool in candidate blocks (bounded memory)."""
    return torch.cat([score(q_sets, q_mask, c_sets[i:i + block], c_mask[i:i + block], kind)
                      for i in range(0, len(c_sets), block)], 1)
