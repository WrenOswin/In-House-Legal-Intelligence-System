# Phase 5: Legal Statute Identification (LSI)

**Task.** Predict the set of applicable statute IDs (0–99) for each case document. Every document has at least
one statute.
**Metrics.** Macro-F1 (primary), micro-F1, subset accuracy.
**Best result.** Mean-probability ensemble of K2 + K5 + K6 + K7: Codabench **0.4035**.
The best single model is K6 (KD student + per-statute attention), at 0.4033.

Full log: [`Phase 5.txt`](Phase%205.txt). Plan: [`Phase5_Plan.md`](Phase5_Plan.md).
Analysis is in [`analysis/`](analysis/):

- `lsi_eda.py` → `data_eda.txt`;
- `analyze.py` → `results.md`, `results.csv`;
- `tfidf_baseline.py`.

## Data

| Split | Documents | Notes |
|---|---|---|
| train | 42,315 | exact duplicates removed from 42,750 |
| val | 10,037 | dev with duplicates of training texts removed |
| test | 13,019 | no labels |

- **Length.** The median document has 430–550 words (about 550–700 BPE tokens). The p90 document needs
  about 8 chunks of 512 tokens.
- **Labels.** 3.7 labels per document on average.
  - The rarest statute still has 272 training documents.
  - The top 20 statutes carry 60% of the label mass.
  - Co-occurrence is strong, e.g. statutes (5, 34) and (12, 13).
- **Statute mentions are masked** as `<SECTION>` and `<ACT>`, and names as `<ENTITY>`. The model must infer
  statutes from the facts. Each mask is mapped to a single existing token (`Section`, `Act`, `X`) instead of
  3–6 BPE pieces.
- **Duplicates.** Train/dev duplicates are removed from val so that the validation score is honest. Short
  boilerplate duplicates often have inconsistent labels.

## Method

The code is in `src/` and reuses the Phase-4 pipeline.

- **Data (`lsi.py`).** Splits, normalisation, memory-mapped tokens, 512-token chunks (at most 4 per document
  in training, 8 evenly spaced at inference), multi-label metrics checked against scikit-learn, threshold
  search and the submission writer.
- **Model (`model.py`).** A Phase-1 backbone produces mean-pooled chunk vectors, which feed 100 independent
  sigmoids. The output bias is initialised to the prior log-odds. There are two heads:
  - `att`: Phase-4 attention pooling over chunks.
  - `label_att`: chunk-level statute logits plus a separate attention over chunks for every statute.
- **Loss.** BCE with positive weights `clip(sqrt(n_neg / n_pos), 1, 10)`. Focal and ASL losses are also
  implemented.
- **Thresholds.** Four strategies are compared:
  - a fixed 0.5;
  - one global threshold;
  - per-label thresholds;
  - per-label thresholds shrunk halfway to the global one.

  Each is scored by **2-fold cross-fitting on val**, the strategy with the best cross-fitted score is chosen,
  and it is refit on all of val. Decoding always outputs at least one statute.
  Unshrunk per-label thresholds look 0.02 better in-sample but gain only 0.003–0.005 cross-fitted.
- **Adaptation (`peft.py`, from scratch).** LoRMA, partial FT, Pfeiffer adapters and full FT.
- **Knowledge distillation.** The teacher is K1 (A d768 LoRMA), with its logits cached for all training
  documents. The loss is `0.5·task + 0.5·T²·BCE(σ(s/T), σ(t/T))`, using tempered sigmoids.
- **Training (`train.py`).** 3 epochs. Checkpoint selection uses the macro-F1 at the best global threshold
  on a fixed 3,000-document val subset. The laptop throttling is the same as in Phase 4.
- **Scripts.**
  - `run_queue.py`: runs K1–K7.
  - `ensemble.py --all-subsets`: scores all 127 subsets.
  - `finalize.py`: rebuilds a submission from a saved checkpoint.
  - `build_cache.py`: tokenises the splits.

## Experiments and submissions (9)

"Val cv" is the honest cross-fitted macro-F1 estimate.

| Folder | Arch / technique | Params (trainable) | Val cv | Codabench |
|---|---|---|---|---|
| `E_ensemble/submission_20261024_212045_K2+K5+K6+K7` | **mean-probability ensemble** | – | 0.4109 | **0.4035** |
| `E_ensemble/submission_20261024_212051_K5+K6+K7` | ensemble of the three A d560 students | – | 0.4089 | 0.4027 |
| `K6_KD_labelatt/submission_20261024_185428_A_d560_full_kd_cbe5af` | A d560 full FT + KD from K1 + `label_att` head | 58.19M (58.19M) | 0.4063 | 0.4033 |
| `K5_KD/submission_20261024_125530_A_d560_full_kd_fd8d35` | A d560 full FT + **KD from K1** | 58.28M (58.28M) | 0.4037 | 0.3978 |
| `K7_A_full_noKD/submission_20261024_211349_A_d560_full_fb9f28` | A d560 full FT, no teacher (KD control) | 58.28M (58.28M) | 0.3902 | 0.3855 |
| `K2_D_partial/submission_20261024_141410_D_d560_partial_633ad4` | D partial FT (top 4/8 blocks) | 54.52M (18.40M) | 0.3850 | 0.3796 |
| `K1_A_lorma/submission_20261018_200321_A_d768_lorma_4c7c72` | A d768 **LoRMA** r=8 (KD teacher) | 142.35M (2.21M) | 0.3789 | 0.3770 |
| `K4_B_full/submission_20261024_163556_B_d512_full_c0537b` | B encoder full FT | 41.75M (41.75M) | 0.3544 | 0.3482 |
| `K3_C_adapter/submission_20261024_154250_C_d560_adapter_30ccdd` | C Pfeiffer adapters (bottleneck 128) | 59.43M (1.36M) | 0.3538 | 0.3472 |

The reference model, which uses no Phase-1 model and was not submitted, is TF-IDF 1–2-grams + one-vs-rest
logistic regression: val cv 0.4082.

## Key observations

1. **Val is a reliable proxy.** With 10,037 val documents, the Pearson correlation between val cv and test is
   0.997. Every submission scores 0.002–0.007 below its cross-fitted estimate.
2. **KD, separated from full fine-tuning.**
   - The KD student K5 beats its K1 teacher by +0.025 cv.
   - About +0.012 of that comes from fully fine-tuning the small model (K7 vs K1).
   - About +0.013 comes from distillation itself (K5 vs K7).
   - On test, KD adds +0.012 (0.3978 vs 0.3855).
3. **Ensembles add almost nothing on test.** The best-of-127 subset gained +0.0046 on val over K6, but only
   +0.0002 on test. This is selection bias.
4. **Rare statutes.** Per-statute F1 rises from 0.31 for the rarest fifth to 0.54 for the most frequent fifth.
   Misses of frequent statutes are confusions within one cluster (5, 34, 39, 43, 97).
5. **Label count.** Documents with one gold statute get 3.2 predictions, and documents with 7+ gold get 6.9
   for 8.7. Count-capped decoding did not help (0.4017 vs 0.4037), so it was not used.
6. **Architecture ranking.** A > D > B ≈ C, as in Phase 1. Full FT rescues the B encoder compared with
   Phase-4 BitFit.
