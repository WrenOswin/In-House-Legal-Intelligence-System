# Phase 4 best model: ensemble of two KD students, J5 + K3

**Codabench macro-F1 0.7771** (submission `E_ensemble/submission_20261021_022358_J5+K3`).
Val macro-F1 0.7958, at a decision threshold of 0.47. The best single model, J5, scores 0.7706.

## Members

| Run | Model | Trainable (backbone + head) | Val | Codabench | Config |
|---|---|---|---|---|---|
| J1 (teacher only) | A d768 + **LoRMA** r=8 | 1.917M + 0.213M | 0.7417 | 0.7431 | `configs/J1_A_lorma_teacher.json` |
| **J5** | A d560, full FT + KD from J1 | 58.07M + 0.156M | 0.7735 | 0.7706 | `configs/J5_KD_member.json` |
| **K3** | C d560 (prefix-LM), full FT + KD from J1 | 58.07M + 0.156M | 0.7847 | 0.7586 | `configs/K3_C_full_KD_member.json` |

The ensemble averages P(label = 1) of J5 and K3 and tunes one threshold on val
(`configs/E_ensemble_best_J5+K3.json`).

## Model (`model.py`, `DocModel`)

1. **Chunks (`cjpe.py`).** Each document is lower-cased, whitespace-normalised and split into 512-token
   chunks. Training takes at most 4 random chunks per document; inference takes 8 evenly spaced chunks.
2. **Encoder.** The Phase-1 backbone encodes each chunk, and its token states are mean-pooled.
3. **Head.** A LayerNorm and a chunk-position feature are applied, and learned attention pooling over the
   chunks produces 2 logits.

## Training (`train.py`)

- Class-balanced cross-entropy, 2 epochs, 32 documents per step, lr 3e-5 for the backbone and 1e-3 for the
  head.
- **Knowledge distillation.** J1's logits for all 31,652 training documents are computed once and cached. The
  student loss is `0.5·CE + 0.5·T²·KL(teacher‖student)` with `T = 2`.
- The checkpoint and threshold are chosen on the 994 clean val documents, evaluated every half epoch. The
  threshold handles the shift from a 41% positive rate in train to 50% in val and test.
- **Data integrity.** The released `test.json` carries labels, and the original `dev.json` contains the test
  documents. Labels are dropped on load, and only the 994 dev documents outside the test set are used for
  validation.

## Files

| File | Role |
|---|---|
| `cjpe.py` | loading, normalisation, memory-mapped chunks, metrics, threshold search, submission writer |
| `backbone.py`, `models_a..d.py`, `bpe.py` | Phase-1 checkpoint and tokenizer loading |
| `peft.py` | LoRMA (J1) and the other PEFT methods |
| `model.py` | `DocModel`: chunk encoder and attention pooling |
| `train.py` | training, KD, evaluation, submission |
| `run_queue.py`, `run.py` | run configurations J1–K4 and the entry point |
| `ensemble.py` | probability averaging, threshold tuning, ensemble submission |

## Reproduce

```bash
cd src
python run_queue.py J1 J5 K3
python ensemble.py J5 K3
```

J1 must run first: it is the KD teacher of J5 and K3, and `run_queue.py` checks that it has finished.
