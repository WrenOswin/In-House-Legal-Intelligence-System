"""
Sentence-sequence labelling on top of a Phase-1 backbone:

  token rows --backbone--> token states --pool over each sentence's span--> sentence vectors [n, d]
  + domain embedding (CL / IT) --> context layer (none | 2-layer BiGRU over the document's sentences)
  --> linear emissions [n, 13] --> CRF (learned label transitions, Viterbi decoding) or softmax.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CRF(nn.Module):
    """Linear-chain CRF over one sequence of sentences. trans[i, j] = score of label i followed by label j.
    No start/end scores: training sequences are random crops of documents, not whole documents."""
    def __init__(self, n_labels, init=None):
        super().__init__()
        t = torch.zeros(n_labels, n_labels) if init is None else torch.as_tensor(init, dtype=torch.float32)
        self.trans = nn.Parameter(t.clone())

    def nll(self, em, y):
        """Mean negative log-likelihood per sentence. em [n, K] float32, y [n] long."""
        n = em.size(0)
        score = em[torch.arange(n, device=em.device), y].sum() + self.trans[y[:-1], y[1:]].sum()
        alpha = em[0]
        for t in range(1, n):
            alpha = torch.logsumexp(alpha[:, None] + self.trans, 0) + em[t]
        return (torch.logsumexp(alpha, 0) - score) / n

    def viterbi(self, em):
        return viterbi(em, self.trans.detach().float().cpu().numpy())

    def marginals(self, em):
        return marginals(em, self.trans.detach().float().cpu().numpy())


def viterbi(em, trans):
    """em [n, K] numpy scores, trans [K, K] -> best label sequence [n]."""
    n, K = em.shape
    score, back = em[0].copy(), np.zeros((n, K), dtype=np.int64)
    for t in range(1, n):
        cand = score[:, None] + trans
        back[t] = cand.argmax(0)
        score = cand.max(0) + em[t]
    path = np.zeros(n, dtype=np.int64)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _lse(x, axis):
    m = x.max(axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis, keepdims=True))).squeeze(axis)


def marginals(em, trans):
    """Per-sentence posterior label probabilities [n, K] of the linear-chain CRF (forward-backward)."""
    n, K = em.shape
    fw, bw = np.zeros((n, K)), np.zeros((n, K))
    fw[0] = em[0]
    for t in range(1, n):
        fw[t] = _lse(fw[t - 1][:, None] + trans, 0) + em[t]
    for t in range(n - 2, -1, -1):
        bw[t] = _lse(trans + (em[t + 1] + bw[t + 1])[None, :], 1)
    lp = fw + bw
    lp -= _lse(lp, 1)[:, None]
    return np.exp(lp)


class RRModel(nn.Module):
    def __init__(self, backbone, n_labels, n_domains=2, context="bigru", gru_hidden=256, gru_layers=2,
                 dropout=0.1, use_crf=True, crf_init=None, pool="mean", domain_emb=True):
        super().__init__()
        self.backbone, self.pool = backbone, pool
        d = backbone.d_model
        self.dom = nn.Embedding(n_domains, d) if domain_emb else None
        if self.dom is not None:
            nn.init.zeros_(self.dom.weight)
        self.drop = nn.Dropout(dropout)
        if context == "bigru":
            self.ctx = nn.GRU(d, gru_hidden, gru_layers, batch_first=True, bidirectional=True,
                              dropout=dropout if gru_layers > 1 else 0.0)
            out_dim = 2 * gru_hidden
        elif context == "none":
            self.ctx, out_dim = None, d
        else:
            raise ValueError(context)
        self.out = nn.Linear(out_dim, n_labels)
        nn.init.normal_(self.out.weight, std=0.02); nn.init.zeros_(self.out.bias)
        self.crf = CRF(n_labels, crf_init) if use_crf else None

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    def encode(self, rows, spans, pad_id, device, token_budget=8192):
        """rows: list of int arrays; spans [n, 3] (row, start, end). Rows are length-sorted and run through
        the backbone in micro-batches of <= token_budget padded tokens. -> sentence vectors [n, d] float32."""
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
        hidden = [None] * len(rows)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and len(rows[order[j]]) * (j + 1 - i) <= token_budget:
                j += 1
            batch = [order[k] for k in range(i, j)]
            T = len(rows[batch[-1]])
            x = np.full((len(batch), T), pad_id, dtype=np.int64)
            for r, k in enumerate(batch):
                x[r, :len(rows[k])] = rows[k]
            x = torch.from_numpy(x).to(device)
            key_pad = x == pad_id if self.backbone.arch in "BC" else None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                h = self.backbone(x, key_pad)
            for r, k in enumerate(batch):
                hidden[k] = h[r, :len(rows[k])]
            i = j
        vecs = []
        for r, s, e in spans.tolist():
            h = hidden[r]
            vecs.append(h[e - 1] if self.pool == "last" else h[s:e].float().mean(0))
        return torch.stack(vecs).float()

    def emissions(self, vecs, domain):
        """vecs [n, d] (one document or one crop of it, in order) -> emissions [n, K] float32."""
        x = vecs
        if self.dom is not None:
            x = x + self.dom.weight[domain]
        x = self.drop(x)
        if self.ctx is not None:
            with torch.autocast("cuda", enabled=False):
                x, _ = self.ctx(x[None].float())
            x = self.drop(x[0])
        return self.out(x).float()

    def loss(self, em, y, class_weight=None, ce_aux=0.0):
        if self.crf is not None:
            l = self.crf.nll(em, y)
            if ce_aux > 0:
                l = l + ce_aux * F.cross_entropy(em, y, weight=class_weight)
            return l
        return F.cross_entropy(em, y, weight=class_weight)

    @torch.no_grad()
    def decode(self, em):
        """-> (labels [n], probabilities [n, K]) as numpy."""
        e = em.float().cpu().numpy()
        if self.crf is not None:
            return self.crf.viterbi(e), self.crf.marginals(e)
        p = torch.softmax(em.float(), -1).cpu().numpy()
        return p.argmax(1), p
