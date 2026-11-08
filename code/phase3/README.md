# Phase 3 best model: D d560 + prefix tuning + BiGRU + CRF (R5)

**Codabench macro-F1 0.4950** (submission `R5_D_prefix/submission_20261012_132235_D_d560_prefix_w_dca8c6`).
Dev macro-F1 0.4098 over all 13 roles (0.4843 over the roles present in dev).

## Model

- **Backbone.** Phase-1 architecture D (efficient Transformer, d560, 8 layers, sliding-window attention with
  GQA), frozen.
- **Prefix tuning (`peft.py`).** 16 trainable prefix vectors per layer, produced by an MLP with hidden size 512.
- **Sentence encoding.** The document is encoded in token windows of the backbone's full context (2,048 for D),
  with 50% overlap at inference. Each sentence is truncated to 128 tokens and its token states are mean-pooled
  into one vector. A domain embedding marks CL vs IT. Training steps take 4 chunks of 64 sentences.
- **Context and decoding (`heads.py`).** A 2-layer BiGRU (hidden 256) runs over the sentence vectors, and a
  linear-chain CRF decodes the role sequence. The CRF transitions are initialised from the training label
  bigrams.
- **Size.** 61.65M parameters in total, 7.34M trainable (2.45M of them in the head).

## Training

AdamW with lr 1e-3 for both prefix and head, weight decay 0.01, 6% warm-up and gradient clipping at 1.0.
Training runs up to 15 epochs with early stopping (patience 4) on dev macro-F1. The best epoch was 13, after
13.1 min with 3.49 GB of VRAM. The full configuration is in
`configs/R5_D_prefix_best.json`.

## Files

| File | Role |
|---|---|
| `rr_data.py` | loading, sentence tokenisation, windows, paths |
| `backbone.py`, `models_a..d.py`, `bpe.py` | Phase-1 checkpoint and tokenizer loading |
| `peft.py` | prefix tuning (and the other PEFT methods) |
| `heads.py` | BiGRU context encoder and CRF |
| `train_rr.py` | training loop, model selection, dev evaluation, submission writer |
| `evaluate.py` | macro-F1 over present and all roles, per-domain scores |
| `run.py`, `run_phase3.py` | command-line entry points |

## Reproduce

```bash
cd src
python run.py '{"exp": "R5_D_prefix", "backbone": "D_d560", "method": "prefix", "step_pause": 0.2}'
```

All other settings are the `DEFAULTS` in `train_rr.py`, which match the saved configuration.
