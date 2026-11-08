# Phase 1: Legal LM Pretraining

**Task.** Continue each of 1,000 test prompts (512 words of legal text) with exactly 128 words.
**Metric.** BERTScore F1 (primary); ROUGE-L and BLEU-4 are secondary.
**Rules.** Own architecture, 50M–300M parameters, trained from scratch, free-GPU-scale compute.
**Best result.** Decoder-only `A_d768` (140.15M parameters, 600M tokens): BERTScore **0.8321**.
This checkpoint is the main backbone for Phases 2–7.

The full log is [`Phase 1.txt`](Phase%201.txt).

## Files

| Notebook | Architecture |
|---|---|
| `Phase1_A_DecoderOnly.ipynb` | A, decoder-only (baseline and final model) |
| `Phase1_A_DecoderOnly_GQA_KVCache.ipynb` | A + grouped-query attention + KV cache + deep-thin shape (final Phase-1 run) |
| `Phase1_B_EncoderDecoder.ipynb` | B, encoder-decoder |
| `Phase1_C_PrefixLM_Hybrid.ipynb` | C, prefix-LM / hybrid |
| `Phase1_D_EfficientTransformer.ipynb` | D, efficient sliding-window Transformer |

Each notebook is self-contained and runs top to bottom through these stages:

1. Config.
2. Download the data (`snapshot_download`).
3. Train or load the BPE tokenizer.
4. Build the token corpus.
5. Build the model and check the parameter count against the 50M–300M bound.
6. Train, with checkpoints every 500 steps and automatic resume.
7. Plot the loss curves.
8. Generate the test continuations and write the submission.

## Architecture

All four models share LLaMA-style components:

- pre-norm RMSNorm;
- rotary position embeddings (RoPE);
- SwiGLU feed-forward layers;
- no bias terms;
- tied input/output embeddings;
- GPT-2-style scaled initialisation;
- PyTorch fused attention (SDPA).

Because of this, the differences between them come from the overall design.

- **A, decoder-only (`DecoderOnlyLM`).** One causal stack: each token sees only earlier tokens.
- **B, encoder-decoder (`EncoderDecoderLM`).** A bidirectional encoder reads a 768-token prompt. A causal
  decoder writes a 256-token continuation, with cross-attention to the encoder in every layer.
- **C, prefix-LM (`PrefixLM`).** One stack. The prompt (prefix) attends bidirectionally, the continuation is
  causal, and a learned segment embedding marks prefix vs continuation.
- **D, efficient (`EfficientLM`).** Causal, with 256-token sliding-window attention in most layers and full
  attention every 4th layer. It uses grouped-query attention (2 K/V heads) and a 2,048-token context.

**Tokenizer.** Byte-pair encoding written by us: 32k vocabulary for the architecture study and 35k for the
final model, trained on 100M characters. The Unicode-aware pre-tokenizer splits numbers into groups of at most
three digits, so section numbers and years stay compositional. Five special tokens are reserved for
downstream inputs: `<|bos|>`, `<|sep|>`, `<|pad|>`, `<|mask|>` and `<|endoftext|>`.

**Data.** The official corpus has more than 4.5B tokens. It is mixed as 35% Acts, 35% Supreme Court,
15% High Court and 15% Tribunal text. The Acts source holds only about 150M tokens, so the data builder
passes any unfilled share of one source to the others. A held-out slice is used for validation.

**Training.** AdamW with β = (0.9, 0.95) and weight decay 0.1. The learning rate warms up over 500 steps,
then follows a cosine decay from 3e-4 to 3e-5. Gradients are clipped at 1.0, training uses bf16, and there is
no dropout.

**Decoding (all scored submissions).** Temperature 0.8, top-k 50, top-p 0.95, and a repetition penalty of
1.1 over the last 256 tokens. Generation produces up to 512 new tokens and the output is truncated to exactly
128 words. The first submission used 320 tokens, which left two outputs short when repetition loops used up
the budget.

## Experiments

### 1. Architecture comparison

All four models were trained on the same budget: 200M tokens, 32k vocabulary and the same data mix.

| Arch | Params | Config | Time | Best val loss (ppl) | Codabench |
|---|---|---|---|---|---|
| A | 58.1M | d560, 8 layers, 8 heads | 133 min | **2.571 (13.1)** | **0.8281** |
| B | 73.0M | d512, 6 enc + 6 dec layers | 49 min | 3.066 (21.4) | 0.8227 |
| C | 58.1M | d560, 8 layers, segment embedding | 153 min | 2.775 (16.0) | 0.8261 |
| D | 54.3M | d560, 8 layers, window 256 | 129 min | 2.830 (16.9) | 0.8270 |

The decoder-only model wins on both validation loss and BERTScore. Our explanation is **learning signal per
token**. A and D learn from every token, C only from the continuation (about 45%), and B from 256 of every
1,024 tokens it reads. D's long-context design gives it nothing here, because prompt plus continuation is only
about 800 tokens. B and C compute loss only on continuation tokens, which are easier to predict, and they are
still worse than A.

### 2. Scaling the decoder-only model

| Run ID | Params | Tokens | Vocab | Time | Best val loss (ppl) |
|---|---|---|---|---|---|
| `d560_L8_T1024_v32k_6613f2` | 58.1M | 200M | 32k | 133 min | 2.571 (13.1) |
| `d640_L8_T1024_v35k_42d59e` | 74.8M | 450M | 35k | 210 min | 2.143 (8.5) |
| `d768_L12_T1024_v35k_5fbdf6` | 140.2M | 600M | 35k | 434 min | **1.928 (6.9)** |
| `d576_L23_T1024_v35k_fbb29e` (GQA + KV cache + deep-thin) | 132.1M | 600M | 35k | 515 min | 1.936 (6.9) |

The d640 run asked for 500M tokens with a 40/40/10/10 mix. The Acts source ran out, which is what prompted
the redistribution logic in the data builder. That run was never uploaded.

### 3. Architectural changes to the decoder-only model

These are in the separate notebook `Phase1_A_DecoderOnly_GQA_KVCache.ipynb`.

- **Grouped-query attention.** Groups of query heads share one key/value head. This gives fewer parameters
  and a smaller KV cache.
- **KV cache.** Keys and values of the prompt are computed once, and each new token runs alone. Generation
  for the test set went from about 2 hours to minutes. Cached and uncached generation give identical tokens
  in fp32.
- **Deep-thin shape (MobileLLM-style).** d576, 23 layers, 9 query heads and 3 K/V heads, at about the same
  parameter budget.

Result: validation loss 1.936 and Codabench 0.8314, against 1.928 and 0.8321 for the best model. That is a tie
at 6% fewer parameters, with 19% longer training. Shape and attention were changed together, so their effects
cannot be separated.

## Submissions

All six scored submissions use the decoding settings above. Each folder in `submissions/<arch>/` holds the
uploaded `submission.zip`, the `prediction.json` and the `gen_config.json` with the decoding settings.

| # | Folder | Model | Val loss | Codabench BERTScore |
|---|---|---|---|---|
| 1 | `A_decoder_only/submission_20260930_231940` | A d560, 200M tokens (max_new_tokens 320) | 2.571 | 0.8281 |
| 2 | `B_encoder_decoder/submission_20261001_115123` | B encoder-decoder, 200M tokens | 3.066 | 0.8227 |
| 3 | `C_prefix_lm/submission_20261001_174934` | C prefix-LM, 200M tokens | 2.775 | 0.8261 |
| 4 | `D_efficient/submission_20261001_232007` | D efficient, 200M tokens | 2.830 | 0.8270 |
| 5 | `A_decoder_only/submission_20261003_185559_d768_L12_T1024_v35k_5fbdf6` | **A d768, 140M, 600M tokens (final model)** | **1.928** | **0.8321** |
| 6 | `A_decoder_only_gqa/submission_20261004_204454_d576_L23_T1024_v35k_fbb29e` | A d576/L23 + GQA + KV cache | 1.936 | 0.8314 |

`A_decoder_only/submission_20261003_012131_d768_L12_T1024_v35k_5fbdf6` is a seventh folder. It uses the same
d768 model with temperature 0.6, top-k 25, top-p 0.9 and repetition penalty 1.5. It was **never uploaded**
and is kept only for reference.

## Key observations

1. **Scale helped only slightly on the metric.** Going from 58M parameters and 200M tokens to 140M and 600M
   tokens halved perplexity (13.1 → 6.9) but raised BERTScore by only 0.004. BERTScore rates any fluent,
   on-topic legal text highly, so the metric compresses large modelling gains.
2. **No overfitting.** Training and validation loss stay close (2.00 vs 1.96 for d768). Validation loss was
   still falling at the end of every run, so more data would help.
3. **Repetition loops.** Strings such as "L.P.(C) L.P.(C) …" or "** ** **" use many tokens per word. Raising
   `max_new_tokens` to 512 fixed outputs that ended short of 128 words.
4. **Windows memory.** When a run needs more than 12 GB, Windows spills into system RAM instead of raising an
   out-of-memory error, and throughput fell from about 23k to 238 tokens/s. The fix was a smaller micro-batch
   with more gradient accumulation.
5. **Noisy validation readings.** Each evaluation averages 200 random windows, so differences below about
   0.05 in validation loss are not meaningful.
