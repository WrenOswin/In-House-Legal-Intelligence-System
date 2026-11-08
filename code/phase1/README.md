# Phase 1 best model: decoder-only A d768

**Codabench BERTScore 0.8321** (submission `A_decoder_only/submission_20261003_185559_d768_L12_T1024_v35k_5fbdf6`).
Validation loss 1.928 (perplexity 6.9). This checkpoint is the main backbone of Phases 2–7.

## Files

- `Phase1_A_DecoderOnly.ipynb` is self-contained and runs top to bottom. Its `CFG` cell holds the exact
  configuration of the best run. The saved outputs are those of that run.
- `gen_config_best.json` holds the decoding settings of the best submission.

## Model

`DecoderOnlyLM` is written from scratch in the notebook:

- 12 layers, 12 heads, d_model 768, context 1,024 tokens: 140.15M parameters;
- pre-norm RMSNorm, RoPE, SwiGLU feed-forward (4 × d_model), no biases, tied embeddings;
- GPT-2-style scaled initialisation, PyTorch SDPA attention.

## Pipeline (notebook stages)

1. **Config.**
2. **Download** the gated dataset with `snapshot_download`.
3. **Tokenizer.** Byte-level BPE written from scratch, with a 35k vocabulary trained on 100M characters. The
   Unicode-aware pre-tokenizer splits numbers into groups of at most three digits. Five special tokens are
   reserved: `<|bos|>`, `<|sep|>`, `<|pad|>`, `<|mask|>` and `<|endoftext|>`.
4. **Token corpus.** 600M training tokens and 2M validation tokens. The mix is 35% Acts, 35% Supreme Court,
   15% High Court and 15% Tribunal. When a source runs out, the builder passes its unfilled share to the others.
5. **Model build** and parameter-count check (50M–300M).
6. **Training.** AdamW with β = (0.9, 0.95) and weight decay 0.1. The learning rate warms up over 500 steps,
   then follows a cosine decay from 3e-4 to 3e-5. Gradients are clipped at 1.0, training uses bf16, and there
   is no dropout. Checkpoints are saved every 500 steps, with automatic resume. Training took 434 min on an
   RTX 5070 Ti laptop GPU.
7. **Loss curves.**
8. **Generation.** Temperature 0.8, top-k 50, top-p 0.95 and a repetition penalty of 1.1 over the last 256
   tokens. Up to 512 new tokens are generated, then truncated to exactly 128 words.

## Run

Open the notebook and run all cells.

- The work directory comes from the environment variable `CAPSTONE_WORK_DIR`; the original was
  `C:/kaggle/working/cs781`.
- Setting `CAPSTONE_SMOKE=1` runs a one-minute end-to-end check with a tiny model.
