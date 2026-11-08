# Phase 3: Rhetorical Role Prediction (RR)

**Task.** Assign one of 13 rhetorical roles to every sentence of a judgment, keeping the order and count of the
sentences. Judgments come from two domains, Corporate Law (CL) and IT. The roles are:

- Fact, Issue;
- ArgumentPetitioner, ArgumentRespondent;
- Statute, PrecedentRelied, PrecedentNotRelied, PrecedentOverruled;
- RatioOfTheDecision, RulingByLowerCourt, RulingByPresentCourt;
- Dissent, None.

**Metric.** Macro-F1.

**Best result.** D (efficient) + prefix tuning with a BiGRU + CRF head: Codabench **0.4950**.
The most robust system is the 10-run ensemble, at 0.4744.

Full log: [`Phase 3.txt`](Phase%203.txt). Plan: [`Phase3_Plan.md`](Phase3_Plan.md).
Tables, figures and error analysis are in [`analysis/`](analysis/).

## Data

|  | Docs | Sentences | Sentences/doc (median) | P(label = previous label) |
|---|---|---|---|---|
| CL train | 40 | 10,593 | 120 | 0.937 |
| IT train | 40 | 6,252 | 126 | 0.754 |
| dev (CL + IT) | 10 | 2,142 | 120–184 | |
| test (CL + IT) | 10 | 2,197 | 114–159 | |

- **Labels come in long runs.** Copying the previous sentence's gold label already scores macro-F1 0.754 on
  dev (all 13 roles: 0.638). Most of the task is finding where a run of one role ends, so a sequence model
  over sentences is essential.
- **Rare roles.** Issue, Dissent (2 CL docs) and PrecedentOverruled (3 sentences) are very rare. Dissent and
  PrecedentOverruled do not appear in dev at all.
- **Two ways to average macro-F1.** The local scorer reports both "present" (labels in gold or prediction) and
  "all-13". Dev all-13 tracked Codabench with no bias (+0.005), while "present" overestimated by 0.074. Models
  were therefore selected on **all-13** from round 2 on.

## Method

The code is in `src/`.

- **Input (`rr_data.py`).** The official sentence lists are used unchanged. In "window mode", sentences are
  packed into the backbone window with newline separators, at most 128 tokens per sentence. The window is
  1,024 tokens, or 2,048 for D. Overlapping windows keep each sentence's copy with the most context.
- **Model (`heads.py`).**
  1. Each sentence vector is the mean of its token states, plus a CL/IT domain embedding.
  2. A two-layer BiGRU (256 units per direction) adds document context.
  3. A linear layer produces emissions.
  4. A linear-chain CRF decodes each document jointly. Its transitions are initialised from the training
     label bigrams.
- **Adaptation (`peft.py`, from scratch).**
  - **LoRMA** (compulsory) and full fine-tuning on A d768.
  - **rsLoRA** on the B encoder.
  - **IA³** on C.
  - **Prefix tuning** on D. MLP-reparameterised key/value prefixes are added inside D's blocked sliding-window
    kernel, so memory is O(T·(2W+P)).
  - LoRMA on D as a method control.
- **Knowledge distillation.** The teacher is R2 (A d768 full FT) and the student A d560. Distillation uses
  sentence-level emissions, so the different tokenizers do not matter. The loss is
  `0.5·CRF-NLL + 0.5·T²·KL(teacher‖student)` with `T = 2`.
- **Training (`train_rr.py`).** Random crops of 64 sentences, 4 crops per step, up to 15–20 epochs, early
  stopping on dev macro-F1. `vram_fraction` 0.75 turns memory overruns into an error instead of a silent
  spill to system RAM.
- **Ensembles (`ensemble.py`).** CRF marginals are averaged and the argmax taken. An optional logit
  adjustment, `argmax log p − τ·log prior`, is applied on top.
- **Driver scripts.**
  - `run_phase3.py`: round 1.
  - `run_round2.py`: method, window and KD-control runs.
  - `run_round3.py`: retraining on train + dev.
  - `sanity.py`: gold round trip and baselines.

All techniques, the CRF and the heads are written from scratch. The only imports are torch, numpy, regex and
the standard library.

## Experiments

### Round 1 (R1–R6, dev all-13 / present)

| Exp | Backbone | Technique | Trainable (backbone + head) | Dev all-13 | Dev present |
|---|---|---|---|---|---|
| R1 | A d768 | **LoRMA** r=8 | 1.92M + 2.77M | 0.4792 | 0.5663 |
| R2 | A d768 | full FT (KD teacher) | 140.1M + 2.77M | 0.4721 | 0.5579 |
| R3 | B encoder | rsLoRA r=8 | 0.49M + 2.37M | 0.3534 | 0.4176 |
| R4 | C | IA³ | 0.03M + 2.45M | 0.4070 | 0.4810 |
| R5 | D | prefix tuning (16 prefixes) | 4.89M + 2.45M | 0.4098 | 0.4843 |
| R6 | A d560 | full FT + KD from R2 | 58.07M + 2.45M | 0.4648 | 0.5493 |

### Round 2

- R7: LoRMA on D, with 2,048- vs 1,024-token windows.
- R8: R5 with 128-sentence crops.
- R1 rerun with a 20-epoch schedule.
- **KD control:** the R6 student trained without a teacher.
- Two ensembles.

### Round 3

R1, R5 and the KD control were retrained on train + dev (90 docs) and ensembled.

### Round 4

Logit adjustment (τ = 0.5) on ensembles and on single models.

## Submissions (22, all in `submissions/`)

| Folder | Description | Dev all-13 | Codabench |
|---|---|---|---|
| `R1_A_lorma/submission_20261011_165805_A_d768_lorma_w_ba1538` | A d768 LoRMA | 0.4792 | 0.4502 |
| `R1_A_lorma/submission_20261012_213638_A_d768_lorma_w_bc8045` | A d768 LoRMA, 20-epoch schedule | 0.4493 | 0.4743 |
| `R1_A_lorma/submission_20261012_225830_A_d768_lorma_w_all_aea4b0` | A d768 LoRMA, trained on train + dev | – | 0.4678 |
| `R2_A_full/submission_20261011_170558_A_d768_full_w_4ef131` | A d768 full FT | 0.4721 | 0.4146 |
| `R3_B_rslora/submission_20261011_170928_B_d512_rslora_w_2d2f72` | B encoder rsLoRA | 0.3534 | 0.3642 |
| `R4_C_ia3/submission_20261011_171823_C_d560_ia3_w_eb3e89` | C IA³ | 0.4070 | 0.4349 |
| `R5_D_prefix/submission_20261012_132235_D_d560_prefix_w_dca8c6` | **D prefix tuning** | 0.4098 | **0.4950** |
| `R5_D_prefix/submission_20261012_224321_D_d560_prefix_w_all_bc6a76` | D prefix, trained on train + dev | – | 0.3976 |
| `R6_KD/submission_20261012_133549_A_d560_full_w_kd_86811a` | A d560 KD student | 0.4648 | 0.4551 |
| `R6_KD/submission_20261012_215157_A_d560_full_w_c030bd` | same student, no teacher (KD control) | 0.4309 | 0.4660 |
| `R6_KD/submission_20261012_231220_A_d560_full_w_all_324fae` | KD control, trained on train + dev | – | 0.4230 |
| `R7_D_lorma/submission_20261012_205747_D_d560_lorma_w_c8f1dd` | D LoRMA, 2,048 window | 0.4493 | 0.4441 |
| `R7_D_lorma/submission_20261012_210827_D_d560_lorma_w_5232ad` | D LoRMA, 1,024 window | 0.4393 | 0.4536 |
| `R8_D_prefix_long/submission_20261012_212529_D_d560_prefix_w_b8d31c` | D prefix, 128-sentence crops | 0.4176 | 0.4218 |
| `E_ensemble/submission_20261012_215232_D4_ensemble` | R5 + R7 ×2 + R8 | 0.4447 | 0.4306 |
| `E_ensemble/submission_20261012_215241_all10_ensemble` | all 10 dev-selected runs | 0.4748 | 0.4744 |
| `E_ensemble/submission_20261012_231238_full3_ensemble` | the three train + dev runs | – | 0.4529 |
| `E_ensemble/submission_20261012_231238_all10_full3_ensemble` | all10 + full3 | – | 0.4697 |
| `E_ensemble/submission_20261013_121454_all10_tau05` | all10, logit adjustment τ = 0.5 | 0.4836 | 0.4548 |
| `E_ensemble/submission_20261013_121454_all10_full3_tau05` | all10 + full3, τ = 0.5 | – | 0.4620 |
| `E_ensemble/submission_20261013_121454_R1all_tau05` | R1 train + dev, τ = 0.5 | – | 0.4759 |
| `E_ensemble/submission_20261013_121454_R5_tau05` | R5, τ = 0.5, argmax of marginals | 0.4001 | 0.4808 |

## Key observations

1. **Small dev and test sets dominate everything.** With 10 dev and 10 test documents, the rank correlation
   between dev and test was only 0.23. Single models differ by up to 0.085 between dev and test. Differences
   below about 0.05 cannot be resolved.
2. **R5's win is partly luck.**
   - It was 4th on dev.
   - Retraining it on 12% more data dropped it to 0.398.
   - The long-context explanation was tested and rejected: D at 1,024 tokens scored 0.454 vs 0.444 at 2,048.
3. **Consistent findings.**
   - LoRMA beats full fine-tuning of the same backbone on test (0.450 vs 0.415), as in Phase 2.
   - The architecture gap holds: A/C/D score 0.44–0.47, B scores 0.36.
4. **KD.** On dev the student gains +0.034 over the no-teacher control. On test the order flips
   (0.455 vs 0.466). Both gaps are inside the noise.
5. **Changes that did not help on test.**
   - Retraining on train + dev.
   - Logit adjustment: −0.009 on average, although it gained +0.01 on dev.
   - Longer crops.
   - The D-only ensemble.
6. **Error analysis.** The models fall back on the majority role:
   - RatioOfTheDecision is predicted 710 times for 403 gold sentences.
   - RulingByLowerCourt, None and Issue are under-predicted.
   - The main confusions are petitioner vs respondent arguments, and facts vs ratio.
7. **Crash.** The first prefix-tuning version used a dense `[2048 × 2064]` mask in every layer and crashed the
   laptop twice. The blocked-kernel rewrite gives identical output (max abs diff 5e-16) at 3.4 GB.
