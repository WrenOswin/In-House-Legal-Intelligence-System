# Phase 6 best model: A d768 + LoRMA r=16 (S1)

**Codabench BERTScore 0.8424** (submission `S1_A_lorma/submission_20261025_190609_A_d768_lorma_224697`).
Val ROUGE-1 / 2 / L 0.472 / 0.238 / 0.253 on 300 held-out documents, with 382 words per summary on average.

## Model

- **Backbone.** Phase-1 `A_d768` (140M), used as a conditional decoder on
  `<|bos|> source <|sep|> summary <|endoftext|>`. The loss is computed only on summary tokens, and the
  pretrained causal attention is kept.
- **Longer window (`seq2seq.py`).** RoPE position interpolation (scale 2.0) doubles the window from 1,024 to
  2,048 tokens: 1,533 source tokens and up to 512 summary tokens.
- **LoRMA (`peft.py`).** Multiplicative low-rank adaptation (Bihany et al., 2025), r = 16 and α = 32, on all 48
  linear layers (qkv, q, kv, proj and SwiGLU). It trains 3.83M of 143.98M parameters.
- **Source selection (`summ.py`, `Salience`).** A ridge regression predicts each sentence's salience from
  hashed uni- and bigram TF-IDF, position bins, first/last-sentence flags and length. The top sentences are
  kept in document order until the 1,533-token budget is full. This step uses scikit-learn; it is an input
  heuristic, not part of the neural model.

## Training (`train.py`)

- AdamW with lr 5e-4, weight decay 0.01, a cosine schedule with 5% warm-up and gradient clipping at 1.0.
- 16 documents per step, micro-batches of at most 4,096 tokens, bf16, up to 3 epochs.
- The model is selected by teacher-forced val loss, evaluated twice per epoch. Training took 44.9 min with
  4.27 GB of VRAM.

## Decoding

Batched KV-cache greedy search with no-repeat-4-gram, 200–512 new tokens and no repetition penalty. The
100 test summaries took 92 s. The full configuration is in `configs/S1_A_lorma_best.json`.

## Files

| File | Role |
|---|---|
| `summ.py` | data, sentence splitting, salience model, source selection, ROUGE, submission writer |
| `build_cache.py` | tokenises every split once, sentence by sentence, with both Phase-1 tokenizers |
| `seq2seq.py`, `models_a..d.py`, `bpe.py` | Phase-1 checkpoint loading, RoPE interpolation, KV-cache generation |
| `peft.py` | LoRMA (and the other PEFT methods) |
| `train.py` | training, val loss and ROUGE, test generation |
| `run_queue.py`, `run.py` | run configurations S1–S7 and the entry point |
| `teacher_gen.py` | teacher summaries for sequence-level KD (used by S5, not by S1) |

## Reproduce

```bash
cd src
python build_cache.py
python run_queue.py S1
```

All other settings are the `DEFAULTS` in `train.py`, which match the saved configuration.
