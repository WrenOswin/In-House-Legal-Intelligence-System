# Phase 6: Abstractive Summarization

**Task.** Write an abstractive summary for each of the 100 test judgments.
**Metrics.** BERTScore F1 (primary), ROUGE-1 and ROUGE-L.
**Best result.** A d768 + LoRMA (S1): Codabench **0.8424**.

Full log: [`Phase 6.txt`](Phase%206.txt). Plan: [`Phase6_Plan.md`](Phase6_Plan.md).
Analysis is in [`analysis/`](analysis/):

- `summ_eda.py` → `data_eda.txt`;
- `analyze.py` → `results.md`, `results.csv`.

## Data

| Split | Documents | Notes |
|---|---|---|
| train | 6,728 | 2 empty summaries removed |
| val | 300 | held-out training documents (seed 6); there is no official dev split |
| test | 100 | |

- **Long documents.** The median judgment has 3.1k words (about 3.7k tokens) and the median summary 640 words.
  Neither fits a 1,024- or 2,048-token window together.
- **The summaries are abstractive.** About a third of reference bigrams do not occur in the source.
- **Salience is flat after the opening.** Salience is the share of a sentence's content words that reappear
  in the summary. The first 10% of a judgment is the least used part, and the rest matters about equally.
- **Truncation loses content.** The first ≈1.3k tokens contain only 57% of the reference's content words.
  Importance-based selection raises that to 64% at the same budget.
- **Extractive references on val (ROUGE-1 / 2 / L).**
  - Lead-500 words: 0.464 / 0.224 / 0.230.
  - Selected-500 words: 0.540 / 0.310 / 0.303.

## Method

The code is in `src/`.

### Input format (`seq2seq.py`)

- **A, C and D** are conditional decoders on `<|bos|> source <|sep|> summary <|endoftext|>`, with the loss
  only on summary tokens.
  - A and D keep the causal attention they were pretrained with.
  - C attends bidirectionally over the source and adds a segment embedding.
  - D keeps its sliding-window and global layers and its native 2,048-token window.
- **B** runs as **Fusion-in-Decoder**. Three 512-token source chunks are encoded separately, and the decoder
  cross-attends to all of them.
- **Longer windows.** RoPE position interpolation doubles A's and C's window from 1,024 to 2,048 tokens:
  1,533 source tokens and 512 summary tokens. B's decoder window goes from 256 to 512.
- **Verification (`check_seq2seq.py`).** The forward passes were re-implemented. The check confirms that they
  reproduce the original Phase-1 losses exactly, and that batched KV-cache greedy decoding equals naive greedy
  decoding in fp32.

### Source selection (`summ.py`, `Salience`)

Instead of chunking and merging, the most useful part of the source is selected.

1. A ridge regression predicts each sentence's salience. Its features are hashed uni- and bigram TF-IDF,
   position bins, first/last-sentence flags and length.
2. The top sentences are kept in document order until the source budget is full.

This regression uses scikit-learn (`HashingVectorizer`, `TfidfTransformer`, `Ridge`). It is an input-selection
heuristic, not part of a neural model. ROUGE is computed with `rouge-score`.

### Adaptation (`peft.py`, from scratch)

LoRMA, LoRA, Houlsby adapters, Pfeiffer adapters and full FT.

### Knowledge distillation (`teacher_gen.py`)

The 140M teacher (35k BPE) and the 58M student (32k BPE) do not share a tokenizer, so **sequence-level KD**
(Kim & Rush, 2016) is used:

1. S1 summarises every training document.
2. For each example, the student's target is the teacher summary with probability 0.5, and otherwise the
   reference.

### Training (`train.py`)

- AdamW, cosine schedule with 5% warm-up, 16 documents per step, micro-batches of at most 4,096 tokens, bf16.
- Up to 3 epochs. The model is selected by teacher-forced val loss, evaluated twice per epoch.
- Learning rates come from short pilots on 800 documents: 5e-4 for LoRMA, 1e-3 for LoRA and adapters, 1e-4 for
  full FT.

### Decoding

Batched KV-cache greedy search with no-repeat-4-gram. Summaries have 200–512 new tokens. There is no
post-training step.

### Scripts

`run_queue.py` runs S1–S7. `decode_sweep.py` is a decoding study and was not used for any submission.

## Experiments and submissions (7)

Val is the ROUGE F1 on 300 held-out documents. There is no local BERTScore, because it needs a pretrained
RoBERTa.

| Folder | Arch / technique | Params (trainable) | Val R-1 / R-2 / R-L | Words | Codabench |
|---|---|---|---|---|---|
| `S1_A_lorma/submission_20261025_190609_A_d768_lorma_224697` | A d768 **LoRMA** r=16 (KD teacher) | 143.98M (3.83M) | .472 / .238 / .253 | 382 | **0.8424** |
| `S6_A_full_noKD/submission_20261026_004518_A_d560_full_d6f92e` | A d560 full FT on references (KD control) | 58.07M (58.07M) | .459 / .224 / .246 | 385 | 0.8379 |
| `S5_KD/submission_20261025_214229_A_d560_full_kd_89ff30` | A d560 full FT + **sequence-level KD** from S1 | 58.07M (58.07M) | .426 / .203 / .237 | 350 | 0.8346 |
| `S3_C_houlsby/submission_20261025_232143_C_d560_houlsby_5daabe` | C Houlsby adapters (bottleneck 128) | 60.38M (2.31M) | .431 / .193 / .230 | 397 | 0.8309 |
| `S2_D_lora/submission_20261025_222320_D_d560_lora_7d36cd` | D LoRA r=16, selected source | 55.70M (1.40M) | .388 / .147 / .215 | 358 | 0.8246 |
| `S7_D_lora_head/submission_20261026_012617_D_d560_lora_65102e` | D LoRA r=16, first-tokens source (ablation of S2) | 55.70M (1.40M) | .387 / .144 / .215 | 365 | 0.8229 |
| `S4_B_fid_pfeiffer/submission_20261025_234458_B_d512_pfeiffer_da0246` | B FiD 3×512 + Pfeiffer adapters | 74.60M (1.58M) | .374 / .123 / .210 | 430 | 0.8162 |

Each folder holds `submission.zip`, `prediction.json` (100 IDs, one non-empty summary each) and `info.json`.

## Key observations

1. **Val ROUGE-L orders all seven systems exactly as test BERTScore does.**
   BERTScore compresses differences: the spread is 0.016, against 0.098 for ROUGE-1.
2. **LoRMA on the 140M model is best** on every metric. It beats the lead-500 extractive baseline with fewer
   words. The architecture order is A > C > D > B, as in Phase 1.
3. **Sequence-level KD hurt.** S5 scored 0.8346 against 0.8379 for the same student without a teacher.
   - Teacher summaries are shorter (median 415 vs 639 words).
   - 31% of the numbers in teacher summaries are not in the source, and the student inherited them.
   - Soft-label KD helped in Phases 4–5. Hard sequences transmit the teacher's mistakes instead of its
     uncertainty.
4. **Source selection vs plain truncation is a tie** (S2 0.8246 vs S7 0.8229), despite +7 points of reference
   coverage in the input. The bottleneck is the model, not the input.
5. **Faithfulness follows quality.** In the best model's summaries, 31% of numbers do not occur in the source.
   In B's summaries the figure is 79%. The "abstractive" text of weak models is mostly hallucination.
6. **Long references are hardest.** References above about 800 words lose 0.12 ROUGE-1, because outputs are
   capped at 512 tokens.
