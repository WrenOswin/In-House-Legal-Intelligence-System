# Phase 5 – Legal Statute Identification: Experiment Plan

Window: **22–26 Oct 2026** (deadline 26 Oct, 23:59 IST). Task: predict the **set** of applicable statute IDs (0–99) for each document. Primary metric: **macro-F1** over the 100 statutes. Secondary metrics: micro-F1 and subset accuracy. Output format: `[{"id": ..., "pred": [ints]}]`, with one object per test doc, unique and valid IDs only.

Rules (capstone doc §11.2): start from the Phase 1 models and use **all four architectures**. Run **≥ 4 substantially different experiments**: ≥ 3 different FT/PEFT techniques (**LoRMA compulsory**) plus ≥ 1 **knowledge distillation** run (a larger teacher distilled into a smaller student). Hyperparameter-only changes don't count as new experiments. Make ≥ 6 valid submissions (README), at most 10 per day.

## 1. Data (details: `Phase 5.txt` §2, `analysis/data_eda.txt`)
| | docs (after de-dup) | Phase-1 BPE tokens (35k vocab) median / p90 | 512-chunks/doc mean | ≤ 8 chunks |
|---|---|---|---|---|
| train | 42,315 (of 42,750) | 554 / 3,611 | 3.39 | 92% |
| val = dev − train duplicates | 10,037 (of 10,181) | 709 / 3,899 | 3.76 | 91% |
| test | 13,019 | 688 / 3,687 | 3.64 | 92% |

- **100 labels**, all present in train and dev. Each doc has 3.7 labels on average (median 3, max 31) and at least 1. The rarest statute still has 272 train / 68 dev docs. Dev has ~0.25× the train count for every label, so **dev is a reliable macro-F1 proxy** (unlike Phase 3).
- **Statute references are masked** (`<SECTION>`, `<ACT>`), so the model must infer statutes from facts and procedure. Unmasked numbers survive in only ~3% of docs.
- **Strong label co-occurrence**, e.g. (5,34), (12,13), (43,97), (13,14).
- **No test-label leak.** Exact duplicates: train–test 109, dev–test 62 (mostly short boilerplate, and their labels often disagree).

**Pre-processing:** whitespace is collapsed and case is kept. Masks are mapped to single BPE tokens: `<SECTION>`→`Section`, `<ACT>`→`Act`, `<ENTITY>`→`X`. Raw `<ENTITY>` costs 6 tokens and appears ~56× per doc, so this cuts ~300 padding tokens from a ~550-token median doc. Val drops the 81 dev docs whose text is also in train, plus duplicates within dev.

## 2. Model (same skeleton for every run, adapted from Phase 4)
Phase-1 backbone + adaptation, then 512-token chunks (≤ 4 random per doc in training, ≤ 8 evenly spaced at inference), then mean-pooled chunk vectors (+ chunk position), then a head with **100 sigmoid outputs**:
- `att`: one attention pooling over chunks, then a linear layer to 100 logits (the Phase 4 head).
- `label_att`: chunk-level statute logits, aggregated with a **separate attention over chunks for each statute**. This is the spec's "chunk-level predictions followed by document-level aggregation": a statute can be evidenced in any part of the document.
- The output bias starts at the log-odds of each statute's prior.

**Loss:** BCE with per-statute `pos_weight = clip(sqrt(neg/pos), 1, 10)` by default (rare statutes up-weighted). Alternatives: plain BCE, focal loss, ASL (asymmetric loss).

**Thresholds** (`lsi.threshold_report`): 0.5 / best global / per-label / per-label shrunk halfway to global. Per-label thresholds are scored by **2-fold cross-fitting on val** (tune on one half, score the other), so the choice isn't flattered by tuning on the scored data. The best cross-fitted strategy is refit on the full val and applied to test. Every doc gets at least its top statute.

**Model selection** during training uses macro-F1 at the best global threshold on a fixed 3,000-doc val subset. After training, the best checkpoint is scored on the full val.

## 3. Experiments (`src/run_queue.py`)
| | Arch | Technique | lr | Purpose |
|---|---|---|---|---|
| K1 | A d768 (140M) | **LoRMA+ r=8** (compulsory) | 1e-4 | strongest PEFT in Phases 2–4; KD teacher |
| K2 | D d560 (54M) | partial / layer-wise FT (top 4 of 8 blocks) | 5e-5 | D; 2nd best in Phase 4 |
| K3 | C d560 (58M) | Pfeiffer adapters, bottleneck 128 | 5e-4 | C; Phase 4 adapters (b=64, lr 2e-4) under-trained → larger and faster |
| K4 | B d512 enc (42M) | **full fine-tuning** | 5e-5 | B; BitFit failed in Phase 4, full FT is cheap at 42M |
| K5 | A d560 (58M) | **knowledge distillation** from K1 (full-FT student) | 3e-5 | mandatory KD; best in Phase 4 |
| K6 (opt.) | A d560 | KD + `label_att` head | 3e-5 | long-document representation comparison |
| K7 (opt.) | A d560 | full FT without a teacher | 3e-5 | separates the KD effect from full FT (Phase 4 obs. 9) |
| E | — | mean-probability ensembles (`src/ensemble.py`) | — | steady extra submissions |

K1–K5 cover all four architectures and 4 techniques from the list (LoRMA, partial FT, adapters, full FT), plus KD, which meets the requirements. Submissions: K1–K5 plus ensembles/K6/K7 for ≥ 6.

**Ablations without new training (for the report, from saved logits):** threshold strategies (0.5 / global / per-label / shrunk), rare vs frequent statutes (per-label F1 vs train frequency), labels per doc, common misses and confusions, co-occurrence post-processing (e.g. add the partner of a high-confidence pair).

**Cheap ablations if time allows:** loss (BCE vs pos_weight vs ASL) on D, chunk selection (first-k vs evenly spaced vs mask-dense chunks around `Section`/`Act`), eval_chunks 8 vs 16.

## 4. Cost estimate (throttled, duty 0.6, from Phase 4 throughput)
Training reads 37.9M tokens per epoch (≤ 2048/doc). Phase 4 read ~53M per epoch. A full val pass is 13.2M tokens and test is 16.4M.

| Run | tok/s (Phase 4) | per epoch | 3 epochs + evals |
|---|---|---|---|
| K1 A d768 LoRMA | 12.8k | ~50 min | ~2.8 h |
| K2 D partial | 26.4k | ~24 min | ~1.4 h |
| K3 C adapters | 22.9k | ~28 min | ~1.6 h |
| K4 B full FT | ~35k (est.) | ~18 min | ~1.1 h |
| K5 A d560 KD | 15.1k | ~42 min | ~2.4 h (+15 min teacher logits) |

Total for K1–K5 is **~9–10 h**. K5 needs K1. Order: K1 → K5 → K2 → K3 → K4, so the strongest pair is in by the end of 22 Oct (first submissions early, per the Codabench-delay note).

## 5. Risks and checks
- **Constant predictor / collapse:** watch labels/doc (gold mean 3.9) and macro-F1 at the first eval. In Phase 4, LoRMA at 5e-4 collapsed, so K1 uses 1e-4.
- **Laptop safety:** keep duty 0.6, vram_fraction 0.6, token_budget 2048, below-normal priority (as in Phase 4).
- **Codabench macro-F1** may differ from local scores if the evaluator treats empty predictions or absent labels differently. Every doc has ≥ 1 prediction and all 100 labels occur in test-sized data, so this is unlikely to matter.
