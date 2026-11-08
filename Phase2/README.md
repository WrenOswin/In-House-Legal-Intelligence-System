# Phase 2: Legal Named Entity Recognition (L-NER)

**Task.** Predict `[start_char, end_char, LABEL]` spans in each document. There are 12 entity types:

- `APP`, `RESP`, `A.COUNSEL`, `R.COUNSEL`;
- `JUDGE`, `WIT`, `AUTH`, `COURT`;
- `STAT`, `PREC`, `CASENO`, `DATE`.

**Metric.** Strict macro-F1: start, end and label must all match exactly.

**Best result.** Ensemble of three `A_d768` models + party-role rules + name propagation: Codabench **0.5590**.
The best single model is LoRMA + word snapping, at 0.5174.

Full log: [`Phase 2.txt`](Phase%202.txt). Plan: [`Phase2_Plan.md`](Phase2_Plan.md).
Results tables, error analysis and figures are in [`analysis/`](analysis/).

## Data

| | Value |
|---|---|
| Training documents | 70 (folds 1 and 2), 8,393 entities |
| Test documents | 35 (fold 3); doc `1305957` is empty and must still appear with `[]` |
| Document length | median 26.6k characters ≈ 5.6k BPE tokens, so every document needs sliding windows |
| Validation | `data/val_split.json`: 56 train / 14 val documents (seed 1312), chosen so that every label has 17–23% of its mentions in val |

### Findings that shaped the design

- **Label imbalance.** `DATE` has 1,575 mentions and `R.COUNSEL` 184. Macro-F1 weights them equally.
- **Party entities sit in the document header.** The median relative position of `APP`, `RESP`, counsel and
  `JUDGE` is 0.15–0.27.
- **The same string can carry different labels.** "Union of India" is `RESP` 28 times, `APP` 14, `PREC` 4 and
  `AUTH` 4. A gazetteer cannot work.
- **Annotation is consistent within a document.** In the 56 training documents, every standalone repeat of an
  annotated string is annotated, always with the same label. This justifies name propagation.
- **Gold spans and token boundaries.** The Phase-1 pre-tokenizer glued punctuation to words (`SCC 47].`), so
  152 gold span ends fell inside a token. The NER tokenizer therefore splits punctuation into single
  characters. After that, a gold → BIO → span round trip scores **0.9944**, the ceiling of the pipeline.

## Method

The code is in `src/`. The notebooks in `notebooks/` call it with one config per experiment.

- **Tokenization and labels (`ner_data.py`, `bpe.py`).** The text is never modified. The tokenizer returns
  each token's character offsets, gold spans become BIO tags on tokens, and predictions are mapped back
  through the same offsets.
- **Windows.** Documents are processed in windows of the backbone's context length, overlapping by half.
- **Model (`train_ner.py`, `backbone.py`).** A Phase-1 backbone with a linear BIO head (25 tags). Causal
  attention was kept: models transfer best in the attention mode they were pretrained in.
- **Decoding (`ner_data.py`, `postprocess.py`).**
  - BIO-constrained Viterbi gives valid, non-overlapping spans.
  - Word snapping extends a span that starts or ends inside a word to the whole word, then merges overlaps.
    4.5% of raw predictions cut a word, for example "Th" + "ukral".
- **Adaptation (`peft.py`, from scratch).** Full fine-tuning, LoRMA, LoRA, Pfeiffer adapters and partial
  fine-tuning.
- **Knowledge distillation.** The teacher is E1 causal (`A_d768`, 35k BPE) and the student `A_d560` (32k BPE).
  The two tokenizers differ, so the teacher's soft tag distributions are aligned to student tokens by
  character offsets (99.9% of student tokens align).
  The loss is `0.5·CE(gold) + 0.5·T²·KL(teacher‖student)` with `T = 2`.
- **Ensemble (`ensemble.py`).** Per-token tag probabilities of three `A_d768` runs are averaged before decoding.
  The runs are E2 LoRMA, E1 bidirectional and E1 causal, which share a tokenizer.
- **Post-processing (`postprocess.py`).** Both rules were designed and checked on training documents only.
  - `resolve_party_roles`: a party's side in the case title ("… Versus …") fixes the `APP`/`RESP` label of
    that party's later mentions.
  - `propagate_strings`: every standalone repeat of a predicted name gets its majority label. This applies
    only to the name labels `APP`, `RESP`, `A.COUNSEL`, `R.COUNSEL`, `JUDGE` and `WIT`.
- **Training.** Up to 25 epochs with early stopping on val macro-F1. Most runs took 3–7 minutes.

## Experiments

| Exp | Backbone | Technique | Trainable | Val macro-F1 |
|---|---|---|---|---|
| E1 | A d768 (140M) | full fine-tuning, bidirectional | 140.2M (100%) | 0.4728 |
| E1 | A d768 | full fine-tuning, causal (KD teacher) | 140.2M | 0.5048 |
| E2 | A d768 | **LoRMA** r=8 on qkv, proj and SwiGLU, causal | 1.9M (1.4%) | 0.5074 |
| E3 | B encoder (42M) | LoRA r=8 | 0.5M | 0.3198 |
| E4 | C prefix-LM (58M) | Pfeiffer adapters (bottleneck 64) | 0.6M | 0.4030 |
| E5 | D efficient (54M) | partial FT (top 4 of 8 blocks), causal | 18.2M (34%) | 0.4614 |
| E6 | A d560 (58M) | full FT + KD from E1 causal | 58.1M | 0.4771 (0.4510 without a teacher) |
| E7 | ensemble of E1 ×2 + E2 | + party roles + name propagation | – | 0.5891 |

### Experiments that did not help

- **Header context.** Prepending the document's first 256 tokens to every window scored 0.492 for E1 and
  0.487 for LoRMA, against 0.505 and 0.507 without it.
- **Retraining LoRMA on all 70 documents.** Test score 0.4877 vs 0.5090. The shorter learning-rate schedule
  changed the run.

These runs were deleted; their numbers are kept in the log.

## Submissions (11, all in `submissions/`)

Each folder holds `submission.zip`, `prediction.json` and `run_info.json` with the run config and validation
scores.

| Folder | Description | Val | Codabench |
|---|---|---|---|
| `E1_A_full/submission_20261009_125313_A_d768_full_8fcd22` | A d768 full FT, bidirectional | 0.4728 | 0.4807 |
| `E1_A_full/submission_20261009_174412_A_d768_full_5c4900` | A d768 full FT, causal | 0.5048 | 0.4790 |
| `E2_A_lorma/submission_20261009_174435_A_d768_lorma_684310` | A d768 LoRMA r=8, causal | 0.5074 | 0.5090 |
| `E2_A_lorma/submission_20261009_180517_A_d768_lorma_all_afa815` | same, retrained on all 70 docs | – | 0.4877 |
| `E2_A_lorma/submission_20261009_192755_A_d768_lorma_684310_wordsnap` | best LoRMA model + word snapping | 0.5185 | 0.5174 |
| `E3_B_lora/submission_20261009_131211_B_d512_lora_e60be9` | B encoder + LoRA | 0.3198 | 0.3631 |
| `E4_C_adapter/submission_20261009_131741_C_d560_adapter_16dd38` | C + Pfeiffer adapters | 0.4030 | 0.4019 |
| `E5_D_partial/submission_20261009_174452_D_d560_partial_c656f6` | D + partial FT | 0.4614 | 0.4628 |
| `E6_KD/submission_20261009_175450_A_d560_full_kd_35adb4` | A d560 KD student | 0.4771 | 0.4684 |
| `E7_ensemble/submission_20261009_204225_ens3_roles` | ensemble + party roles | 0.5622 | 0.5535 |
| `E7_ensemble/submission_20261009_204219_ens3_roles_propnames` | **ensemble + party roles + name propagation** | 0.5891 | **0.5590** |

Only the submissions marked "word snapping" and the E7 ensembles use word snapping. It was added after the
others were made.

## Key observations

1. **The Phase-1 ranking carries over.** Best per architecture: A 0.5090 > D 0.4628 > C 0.4019 > B 0.3631.
2. **LoRMA beats full fine-tuning on test** (0.5090 vs 0.4790/0.4807) while training 1.4% of the weights.
   With only 56 training documents, freezing the backbone limits overfitting.
3. **KD.** The 58M student scores 98% of its 140M teacher (0.4684 vs 0.4790). On validation, KD added
   +0.026 over the same student without a teacher.
4. **Where the final gain comes from (test):**
   - LoRMA alone: 0.5090.
   - + word snapping: 0.5174 (+0.008).
   - + 3-model ensemble and party roles: 0.5535 (+0.036).
   - + name propagation: 0.5590 (+0.006).

   Name propagation was chosen after looking at validation and gained +0.027 there, but only +0.006 on test.
5. **Error analysis (`analysis/error_analysis/`).**
   - `AUTH` causes half of the false positives: generic words such as "Government" or "Board".
   - Boundary errors are common: 196 spans too short and 121 too long.
   - `APP`/`RESP` stay hardest, because a party's role is stated only in the header.
6. **Validation noise.** The 14-document validation set tracks test within about ±0.03, so smaller
   differences are noise.
