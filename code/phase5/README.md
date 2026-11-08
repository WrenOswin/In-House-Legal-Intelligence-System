# Phase 5 best model: mean-probability ensemble K2 + K5 + K6 + K7

**Codabench macro-F1 0.4035** (submission `E_ensemble/submission_20261024_212045_K2+K5+K6+K7`).
Cross-fitted val macro-F1 0.4109. The best single model, K6, scores 0.4033.

## Members

| Run | Model | Params (trainable) | Val cv | Codabench | Config |
|---|---|---|---|---|---|
| K1 (teacher only) | A d768 + **LoRMA** r=8 | 142.35M (2.21M) | 0.3789 | 0.3770 | `configs/K1_A_lorma_teacher.json` |
| **K2** | D d560, partial FT (top 4 of 8 blocks) | 54.52M (18.40M) | 0.3850 | 0.3796 | `configs/K2_D_partial_member.json` |
| **K5** | A d560, full FT + KD from K1 | 58.28M (58.28M) | 0.4037 | 0.3978 | `configs/K5_KD_member.json` |
| **K6** | A d560, full FT + KD from K1 + per-statute `label_att` head | 58.19M (58.19M) | 0.4063 | 0.4033 | `configs/K6_KD_labelatt_member.json` |
| **K7** | A d560, full FT, no teacher | 58.28M (58.28M) | 0.3902 | 0.3855 | `configs/K7_A_full_noKD_member.json` |

The ensemble averages the four members' sigmoid probabilities. Its threshold strategy was picked by 2-fold
cross-fitting on val: per-label thresholds shrunk halfway to the global one (`label_s0.5`). The 100 thresholds
are in `configs/E_ensemble_best_K2+K5+K6+K7.json`.

## Model (`model.py`)

1. **Chunks (`lsi.py`).** Documents are split into 512-token chunks: at most 4 per document in training and 8
   evenly spaced at inference. The masks `<SECTION>`, `<ACT>` and `<ENTITY>` are mapped to single existing
   tokens.
2. **Encoder.** The Phase-1 backbone produces a mean-pooled vector per chunk.
3. **Heads.** 100 independent sigmoids, with the output bias initialised to the prior log-odds.
   - `att` (K2, K5, K7): attention pooling over chunks.
   - `label_att` (K6): chunk-level statute logits plus a separate attention over chunks for every statute.

## Training (`train.py`)

- BCE with positive weights `clip(sqrt(n_neg / n_pos), 1, 10)`, 3 epochs.
- **Knowledge distillation (K5, K6).** K1's logits are cached for all training documents. The loss is
  `0.5·task + 0.5·T²·BCE(σ(s/T), σ(t/T))`, using tempered sigmoids.
- Checkpoints are selected on the macro-F1 at the best global threshold, over a fixed 3,000-document val subset.
- Decoding always outputs at least one statute.

## Files

| File | Role |
|---|---|
| `lsi.py` | splits, de-duplication, memory-mapped tokens, chunks, metrics, threshold strategies, submission |
| `build_cache.py` | tokenises every split once for both Phase-1 tokenizers |
| `backbone.py`, `models_a..d.py`, `bpe.py` | Phase-1 checkpoint and tokenizer loading |
| `peft.py` | LoRMA (K1), partial FT (K2) and the other PEFT methods |
| `model.py` | chunk encoder with the `att` and `label_att` heads |
| `train.py` | training, KD, evaluation, submission |
| `run_queue.py`, `run.py` | run configurations K1–K7 and the entry point |
| `finalize.py` | rebuilds a submission from a saved checkpoint after an interrupted run |
| `ensemble.py` | probability averaging, cross-fitted threshold choice, ensemble submission |

## Reproduce

```bash
cd src
python build_cache.py
python run_queue.py K1 K2 K5 K6 K7
python ensemble.py K2 K5 K6 K7
```

K1 must run first: it is the KD teacher of K5 and K6.
