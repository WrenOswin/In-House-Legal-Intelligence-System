"""Sanity checks for seq2seq.py (run once after changes):
  1. teacher-forced summary logits == the original Phase-1 model's logits (rope_scale 1, no adaptation)
  2. batched, left-padded KV-cache greedy generation == step-by-step greedy with the original model
"""
import numpy as np
import torch

import summ
from seq2seq import load_summarizer

dev = "cuda"
torch.manual_seed(0)
for name in ["A_d560", "C_d560", "D_d560", "B_d512"]:
    m, tok, info = load_summarizer(name, max_len=1024 if name != "D_d560" else 2048, rope_scale=1.0, device=dev)
    m.eval()
    lm = m.lm
    data = summ.load_tokens(info["tokenizer"], "val")[:3]
    srcs = [summ.context(d, [300, 180, 420][i], "head") for i, d in enumerate(data)]
    tgts = [d["summary"][:40] for d in data]
    with torch.no_grad():
        # 1. teacher forcing vs original forward
        if m.arch == "B":
            src = torch.tensor(np.asarray(srcs[0][:300]), device=dev)[None].long()
            t = [m.bos] + list(tgts[0]) + [m.eot]
            tin = torch.tensor(t[:-1], device=dev)[None]
            ref, _ = lm(src, tin)
            loss_ref = torch.nn.functional.cross_entropy(ref[0].float(), torch.tensor(t[1:], device=dev))
            loss, n = m.loss([(([srcs[0][:300]]), tgts[0], True)])
        else:
            s = [m.bos] + list(srcs[0]) + [m.sep]
            full = torch.tensor(s + list(tgts[0]) + [m.eot], device=dev)[None]
            if m.arch == "C":
                ref, _ = lm(full, torch.tensor([len(s)], device=dev))
            else:
                ref, _ = lm(full)
            loss_ref = torch.nn.functional.cross_entropy(ref[0, len(s) - 1:-1].float(), full[0, len(s):])
            loss, n = m.loss([(srcs[0], tgts[0], True)])
        print(f"{name}: teacher-forced loss mine {loss.item():.5f} vs original {loss_ref.item():.5f}")
        # 2. batched cached generation vs naive greedy (no penalties)
        gen_in = [[s[:200]] for s in srcs] if m.arch == "B" else srcs
        outs = m.generate(gen_in, max_new=24)
        for b in range(3):
            if m.arch == "B":
                src = torch.tensor(np.asarray(srcs[b][:200]), device=dev)[None].long()
                y = [m.bos]
                for _ in range(24):
                    lg, _ = lm(src, torch.tensor(y, device=dev)[None])
                    lg = lg[0, -1].float(); lg[m.banned] = -1e9
                    y.append(int(lg.argmax()))
                    if y[-1] == m.eot:
                        break
                naive = [i for i in y[1:] if i != m.eot]
            else:
                s = [m.bos] + list(srcs[b]) + [m.sep]
                y = list(s)
                for _ in range(24):
                    x = torch.tensor(y, device=dev)[None]
                    lg, _ = lm(x, torch.tensor([len(s)], device=dev)) if m.arch == "C" else lm(x)
                    lg = lg[0, -1].float(); lg[m.banned] = -1e9
                    y.append(int(lg.argmax()))
                    if y[-1] == m.eot:
                        break
                naive = [i for i in y[len(s):] if i != m.eot]
            same = sum(a == c for a, c in zip(outs[b], naive))
            print(f"   gen doc {b}: cached==naive on {same}/{len(naive)} tokens | {tok.decode(outs[b])[:90]!r}")
    del m
    torch.cuda.empty_cache()
