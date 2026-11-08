# Phase 4: Court Judgment Prediction (CJPE / ILDC)

**Task.** Predict the binary outcome (0/1) of each case from its judgment text, with the final decision
removed.
**Metric.** Macro-F1.
**Best result.** Ensemble of the A d560 KD student (J5) and C with full FT + KD (K3): Codabench **0.7771**.
The best single model is J5, at 0.7706.

Full log: [`Phase 4.txt`](Phase%204.txt).

## Data

| Split | Documents | Notes |
|---|---|---|
| train | 31,652 unique (37,387 rows) | 41% positive, mostly 1990–2020, cased |
| val | 994 | dev documents outside the test set; 50% positive, 1950–90, lower-cased |
| test | 1,517 | |

- **Long documents.** The median is about 3.5k BPE tokens, and only 4% of val/test fit in 1,024 tokens.
- **Prior and style shift.** Train is 41% positive, while val and test are 50%. Every split is lower-cased and
  whitespace-normalised to remove the case mismatch.
- **Evidence is spread over the document.** A TF-IDF + logistic-regression reference on val scores 0.690 on
  the whole document, 0.640 on the last 512 words and 0.593 on the first 512 words. This motivates a chunked
  document model.
- **Label leak (never used).** The released `test.json` carried labels, and the original `dev.json` contained
  all 1,517 test documents with their labels. Labels are dropped on load, and validation uses only the 994
  clean dev documents. The TAs' later fix of `dev.json` gave exactly these 994 documents.

## Method

The code is in `src/`. It is a lightweight pipeline that keeps the laptop usable during training.

- **Chunks (`cjpe.py`).** Documents are split into 512-token chunks: at most 4 random chunks per document in
  training and 8 evenly spaced chunks at inference. Tokens are memory-mapped.
- **Model (`model.py`, `DocModel`).**
  1. The Phase-1 backbone encodes each chunk, and its token states are mean-pooled.
  2. A LayerNorm and a chunk-position feature are applied.
  3. Learned attention pooling over chunks produces 2 logits.
- **Loss and threshold.** Class-balanced cross-entropy. The decision threshold is tuned on val, which handles
  the 41% → 50% prior shift.
- **Adaptation (`peft.py`, from scratch).**
  - LoRMA.
  - BitFit: zero-initialised biases are added to every block's linear layers, because the backbones have none.
  - Pfeiffer adapters.
  - Partial FT (top 4 of 8 blocks + final norm).
  - Full FT.
- **Knowledge distillation (`train.py`).** The teacher is J1 (A d768 LoRMA). Its logits for all 31,652
  training documents are computed once and cached. The loss is
  `0.5·CE + 0.5·T²·KL(teacher‖student)` with `T = 2`, on document-level logits.
- **Training.** 2 epochs, 32 documents per step, head learning rate 1e-3. The checkpoint and threshold are
  chosen on val, which is evaluated every half epoch.
- **Laptop-comfort settings.** GPU duty cycle 0.6, below-normal priority, VRAM fraction 0.6 and a 2,048-token
  budget per micro-batch. These are not part of the run ID.
- **Scripts.**
  - `run_queue.py`: runs J1–J5 and K1–K4. Finished runs are skipped, and the KD teacher is pinned to J1's run.
  - `ensemble.py`: averages P(1) over runs and tunes the threshold on val. `--all-subsets` prints every
    subset's score.

## Experiments and submissions (11)

| Folder | Arch / technique | Trainable (backbone + head) | LR | Val | Codabench |
|---|---|---|---|---|---|
| `E_ensemble/submission_20261021_022358_J5+K3` | **mean P(1) of J5 and K3** (threshold 0.47) | – | – | 0.7958 | **0.7771** |
| `J5_KD/submission_20261017_182836_A_d560_full_kd_a06074` | A d560 full FT + **KD from J1** | 58.07M + 0.156M | 3e-5 | 0.7735 | 0.7706 |
| `K1_A_full_noKD/submission_20261020_214500_A_d560_full_c2d7b3` | A d560 full FT, no teacher (KD control) | 58.07M + 0.156M | 3e-5 | 0.7625 | 0.7618 |
| `K3_C_full_KD/submission_20261021_012327_C_d560_full_kd_772baf` | C full FT + KD from J1 | 58.07M + 0.156M | 3e-5 | 0.7847 | 0.7586 |
| `J1_A_lorma/submission_20261017_124501_A_d768_lorma_09e521` | A d768 **LoRMA** r=8 (KD teacher) | 1.917M + 0.213M | 1e-4 | 0.7417 | 0.7431 |
| `E_ensemble/submission_20261017_155841_J1+J4` | mean P(1) of J1 and J4 | – | – | 0.7486 | 0.7426 |
| `J4_D_partial/submission_20261017_155747_D_d560_partial_029156` | D partial FT (top 4/8 blocks) | 18.194M + 0.156M | 5e-5 | 0.7352 | 0.7052 |
| `K2_D_lorma/submission_20261020_233103_D_d560_lorma_286232` | D LoRMA r=8 (method control vs J4) | 0.824M + 0.156M | 1e-4 | 0.6357 | 0.6288 |
| `J3_C_adapter/submission_20261017_144820_C_d560_adapter_df05cf` | C Pfeiffer adapters (bottleneck 64) | 0.578M + 0.156M | 2e-4 | 0.6354 | 0.5906 |
| `K4_B_lorma_KD/submission_20261021_022333_B_d512_lorma_kd_42b205` | B encoder LoRMA r=8 + KD from J1 | 0.639M + 0.142M | 2e-4 | 0.6146 | 0.5767 |
| `J2_B_bitfit/submission_20261017_132848_B_d512_bitfit_555843` | B encoder BitFit | 0.040M + 0.142M | 1e-3 | 0.5573 | 0.5273 |

Each folder holds `submission.zip`, `prediction.json` (1,517 IDs in `test.json` order, int 0/1) and
`run_info.json`.

## Key observations

1. **The KD student beats its teacher.**
   - On test, J5 scores 0.7706 against J1's 0.7431.
   - It uses 43% of the training memory (1.36 vs 3.19 GB) and 60% of the inference time.
   - The no-teacher control K1 scores 0.7618. Part of the gain therefore comes from fully fine-tuning the
     smaller model, and part (+0.009 on test) from the teacher's soft labels.
   - KD mainly speeds up learning: after half an epoch, the student with a teacher is at 0.707 and without one
     at 0.618.
2. **Technique matters a lot with 31k training documents.**
   - Full FT, partial FT and LoRMA on A all work.
   - BitFit stays at chance-level loss (0.687–0.697).
   - On D, partial FT beats LoRMA by 0.10.
   - Full FT + KD lifts C from 0.635 (adapters) to 0.785.
3. **Val predicts test well.** With 994 val documents, val ranks the round-1 submissions in the same order as
   Codabench. Test − val ranges from −0.045 to +0.001.
4. **Ensembles.** Two strong models of different architectures (J5 + K3) gain +0.007 on test. J1 + J4 gained
   nothing.
5. **LoRMA learning rate.** At 5e-4 it collapsed into a constant predictor within one epoch. At 1e-4 with
   32 documents per step it trained steadily.
6. **Frozen vs fine-tuned.** Frozen Phase-1 features with a probe head reach only 0.55–0.62, below TF-IDF
   (0.69). The backbone itself has to be adapted.
