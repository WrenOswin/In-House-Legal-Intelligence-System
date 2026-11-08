# Phase 7 best model: A d768 + LoRMA bi-encoder fused with trigram BM25 (P1)

**Codabench micro-F1@7 0.4312** (submission `P1_A_lorma/submission_20261027_155621_A_d768_lorma_a4ffaa`).
Dev F1@7 0.4569 fused, against 0.3591 for the model alone and 0.4325 for BM25 alone. A perfect ranker scores
0.618 on dev, because many queries do not have exactly 7 gold cases.

## Model

- **BM25 (`pcr.py`).** Written from scratch with `scipy.sparse`, on word trigrams with b = 1.0. Queries quote
  or paraphrase the cases they cite, so exact phrases carry the relevance signal.
- **Bi-encoder (`model.py`, `ChunkEncoder`).**
  1. The Phase-1 `A_d768` backbone produces mean-pooled vectors for 512-token chunks.
  2. A linear projection, initialised to the identity, maps them to unit vectors.
  3. The **maxsim** score averages, over query chunks, each chunk's best cosine match among the candidate's
     chunks.

  Queries and candidates share the encoder. Training uses 4 chunks per case, chosen by BPE-trigram overlap
  with the paired document. Inference uses 16 evenly spaced chunks.
- **LoRMA (`peft.py`).** r = 8 and α = 16 on all 48 linear layers. It trains 2.51M of 142.65M parameters.
- **Fusion.** Candidates are ranked by `z(BM25) + w·z(model)` with w = 0.15, chosen on dev. Each query's own
  document is masked from its ranking.

## Training (`train.py`)

- Each step takes 16 training queries, each with 1 random gold case and 1 hard negative from the BM25-trigram
  top 30.
- The loss is multi-positive InfoNCE over all in-batch candidates, with τ = 0.05.
- **GradCache** fits the batch in 12 GB of VRAM. It is exact because dropout is 0.
- 400 steps, with dev evaluated every 100 steps. The best checkpoint was at step 200, after 52.4 min.

The full configuration, including the dev fusion table for every `w`, is in `configs/P1_A_lorma_best.json`.

## Files

| File | Role |
|---|---|
| `pcr.py` | data, masks, BM25, F1@7 metrics, self-retrieval masking, submission writer |
| `backbone.py`, `models_a..d.py`, `bpe.py` | Phase-1 checkpoint and tokenizer loading |
| `peft.py` | LoRMA (and the other PEFT methods) |
| `model.py` | `ChunkEncoder` and maxsim scoring |
| `train.py` | chunk selection, InfoNCE + GradCache training, dev evaluation, fusion, submission |
| `run_queue.py`, `run.py` | run configurations P1–P6 and the entry point |

## Reproduce

```bash
cd src
python run_queue.py P1
```

All other settings are the `DEFAULTS` in `train.py`, which match the saved configuration.
