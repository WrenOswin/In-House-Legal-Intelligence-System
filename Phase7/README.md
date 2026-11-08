# Phase 7: Prior Case Retrieval (PCR)

**Task.** For each test query case, return exactly 7 unique candidate IDs from that query's own pool, best
first.
**Metrics.** Micro-F1@7 (primary), micro-P@7 and micro-R@7. Gold sets are not truncated.
**Best result.** A d768 + LoRMA bi-encoder fused with word-trigram BM25 (P1): Codabench **0.4312**.

Full log: [`Phase 7.txt`](Phase%207.txt).

## Data

| Split | Queries | Candidate pool | Gold per query |
|---|---|---|---|
| train | 827 (817 with gold) | 4,320 | median 4, mean 6.4 |
| dev | 118 | 1,023 | median 5, mean 8.3 (range 1–92) |
| test | 237 | 1,727 | – |

- **Long cases.** Cases are long (median about 4.5k tokens). `<CITATION>` marks where the cited prior cases
  were removed.
- **Masks.** `<NAME>`, `<ORG>` and `<CITATION>` are mapped to single tokens.
- **Self-retrieval.** Every query is also a document of its own pool, so a query must never retrieve itself.
  `pcr.rank()` masks it.
- **Metric ceiling.** Many queries have fewer or more than 7 gold cases, so a perfect ranker scores only
  **0.618** on dev.

## Method

The code is in `src/`.

- **BM25 (`pcr.py`).** Written from scratch with `scipy.sparse`. A sweep over term types on dev gave:

  | Terms | Dev F1@7 |
  |---|---|
  | words | 0.09 |
  | bigrams | 0.386 |
  | trigrams (b = 1.0) | **0.4325** |
  | 4-grams | 0.416 |

  Queries quote or paraphrase passages of the cases they cite, so exact phrases carry the relevance signal.
- **Bi-encoder (`model.py`, `ChunkEncoder`).**
  1. A shared encoder (a Phase-1 backbone) produces mean-pooled vectors for 512-token chunks.
  2. A linear projection, initialised to the identity, maps them to unit vectors.
  3. The **maxsim** score averages, over query chunks, each chunk's best cosine match among the candidate's
     chunks.

  Queries and candidates share the encoder and the truncation policy. Training uses 4 chunks per case, chosen
  by BPE-trigram overlap with the paired document. Inference uses 16 evenly spaced chunks.
- **Training (`train.py`).**
  - Each step takes 16 training queries, each with 1 random gold case and 1 hard negative from the BM25-trigram
    top 30.
  - The loss is multi-positive InfoNCE over all in-batch candidates, with τ = 0.05.
  - **GradCache** fits the batch in 12 GB. It is exact because dropout is 0.
  - 400 steps, with dev evaluated every 100 steps.
- **Knowledge distillation (P5).** The loss is `(1−a)·InfoNCE + a·KL(teacher in-batch distribution ‖ student)`.
  The teacher is P1, with its maxsim scores over the full training pool computed once.
- **Submission.** Candidates are ranked by `z(BM25) + w·z(model)`, where `w` and the checkpoint are picked on
  dev. Self-retrieval is masked.
- **Scripts.** `run_queue.py` runs P1–P6. `zeroshot.py` evaluates a frozen Phase-1 backbone as a retriever.

## Experiments and submissions (6)

All runs use the same recipe: 400 steps, 16 queries per step, maxsim, τ = 0.05.

| Folder | Arch / technique | Params (trainable) | Dev F1@7, model alone | Dev F1@7, fused (w) | Codabench |
|---|---|---|---|---|---|
| `P1_A_lorma/submission_20261027_155621_A_d768_lorma_a4ffaa` | A d768 **LoRMA** r=8 (KD teacher) | 142.65M (2.51M) | 0.3591 | 0.4569 (0.15) | **0.4312** |
| `P5_KD/submission_20261027_181757_A_d560_full_kd_127552` | A d560 full FT + **KD from P1** | 58.38M (58.38M) | 0.3469 | 0.4569 (0.15) | 0.4300 |
| `P3_C_ia3/submission_20261027_170945_C_d560_ia3_a16b50` | C **IA³** | 58.41M (0.34M) | 0.3013 | 0.4480 (0.1) | 0.4300 |
| `P4_B_full/submission_20261027_172539_B_d512_full_05685b` | B encoder full FT | 41.82M (41.82M) | 0.2635 | 0.4491 (0.15) | 0.4294 |
| `P6_A_full_noKD/submission_20261027_145852_A_d560_full_6926e7` | A d560 full FT, no teacher (KD control) | 58.38M (58.38M) | 0.3246 | 0.4503 (0.2) | 0.4276 |
| `P2_D_partial/submission_20261027_162934_D_d560_partial_e92733` | D **partial FT** (top 4/8 blocks) | 54.62M (18.51M) | 0.3146 | 0.4503 (0.15) | 0.4229 |

References on dev:

- BM25 trigrams alone: 0.4325.
- Frozen Phase-1 A d768: 0.3091 alone, 0.4447 fused.
- Oracle: 0.618.

Each folder holds `submission.zip`, `prediction.json` (237 queries, 7 unique pool IDs each) and `run_info.json`.
The six submissions differ: the mean overlap between their top-7 lists is 0.87–0.91.

## Key observations

1. **Phase-1 representations transfer.** The frozen A d768 alone (0.309) is already far better than word BM25
   (0.09). Fine-tuning lifts it to 0.359. The architecture ranking is A > D > C > B, as before.
2. **The neural score mainly re-weights BM25.** The best `w` is 0.1–0.2. It lifts recall in the top 20 from
   0.58 to 0.61. A pooled representation captures shared quoted passages poorly.
3. **KD helps a little.** The student scores 0.4300 against 0.4276 for the no-teacher control, which is
   0.0012 behind its 140M teacher.
4. **Dev is optimistic.** Every submission scores 0.018–0.027 below dev, because both `w` and the checkpoint
   were picked on the same 118 dev queries. All six test scores lie within 0.008, which is noise at 237
   queries.
5. **No gains from these.** Bigram or 4-gram BM25 fusion, and ensembles of neural scorers (best 0.4558 on dev).
   None of them was submitted.
