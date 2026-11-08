# Phase 3 – Rhetorical Role Prediction: Experiment Plan

Window: **12–16 Oct 2026** (deadline 16 Oct, 23:59 IST). Metric: **macro-F1** over the 13 rhetorical-role labels (secondary: macro-P, macro-R). Every test sentence gets exactly one label, and sentence order is preserved.

Rules (capstone doc §9): start from the Phase 1 models and use **all four architectures**. Run **≥ 4 substantially different experiments**: ≥ 3 different FT/PEFT techniques (**LoRMA compulsory**) plus ≥ 1 **knowledge distillation** run. Changing only hyperparameters doesn't count as a new experiment. Make ≥ 4 valid submissions (README says 6, so aim for 7–8), at most 10 per day.

## 1. What the data looks like
The source is `C:\kaggle\working\cs781\hf_data\Phase_3_RR` (Train downloaded 2026-10-11). Each document is `{"id", "text": [sentences], "labels": [labels]}`. Test files map `id -> [sentences]`.

| | CL (Corporate Law) | IT (Information Tech.) |
|---|---|---|
| Train | 40 docs, 10,593 sents | 40 docs, 6,252 sents |
| Dev | 5 docs, 1,396 sents | 5 docs, 746 sents |
| Test | 5 docs, 1,339 sents | 5 docs, 858 sents |
| Sents/doc (train) | 19 – **1,816**, median 120 | 25 – 442, median 126 |
| Words/sentence | median 27, p95 72, max 733 | median 28, p95 84, max 1,054 |
| P(label = previous label) | **0.937** | **0.754** |

The 100 doc IDs are unique across domains and splits, so there is one `prediction.json` with 10 keys.

**Train label counts (number of docs containing the label in brackets):**
- CL: Fact 2979, Ratio 2856, PrecRelied 1305, ArgResp 996, ArgPet 976, RulingLower 438, Statute 367, **Dissent 284 (2 docs)**, PrecNotRelied 209, RulingPresent 113, None 53, **Issue 17**
- IT: Ratio 1445, Fact 1119, Statute 732, PrecRelied 640, ArgPet 637, RulingLower 436, PrecNotRelied 360, ArgResp 352, None 244, RulingPresent 194, Issue 90, **PrecOverruled 3 (1 doc)**

What this means for the design:
- **Labels come in long runs**, especially in CL (94% of sentences share the previous label). Document-level context and a transition model (CRF/Viterbi) should give the largest gain, so the "context" question is central.
- **Macro-F1 and rare labels:** Issue (CL), PrecedentOverruled, Dissent and None are tiny. Dissent and PrecOverruled don't appear in dev at all. It isn't yet known whether Codabench averages over all 13 labels or only over the labels present in gold ∪ pred. **Find out before tuning rare-label handling** (ask on Discord). The local scorer reports both variants.
- **The domains differ:** CL is mostly Fact/Ratio/Arguments with long runs. IT has more Statute, None, PrecNotRelied and Issue, with shorter runs. Test this with a domain embedding vs separate per-domain models.
- **Sizes are small:** ~17k training sentences of ~35–40 BPE tokens each, about 0.7M tokens per epoch, similar to Phase 2. Runs take minutes, so many ablations are affordable.
- **Dev and test are tiny (10 docs each):** expect ±0.03 or more noise (Phase 2 obs. 13/19). Use 2 seeds for the runs that matter, and don't trust gaps < 0.03.
- Very long sentences (up to ~1.3k tokens) are truncated to 128 tokens in sentence-only mode. That affects < 1% of sentences.

## 2. Model design (shared by all experiments, so differences come from technique and architecture)
A **hierarchical sentence-sequence labeller:**
1. **Sentence encoder** = Phase 1 backbone + adaptation method. It runs in two context modes:
   - **S – sentence-only:** each sentence is encoded on its own and mean-pooled (or last-token pooled for causal models).
   - **W – in-window context:** contiguous sentences are packed into one backbone window, each preceded by a separator token. The window is 1024 tokens for A/C, 768 for the B encoder and 2048 for D. Each sentence is pooled over its own token span. Windows overlap by ~25%, and each sentence keeps the most-central copy. For causal models this gives every sentence its left context inside the backbone. Attention stays in its pretrained mode (Phase 2 obs. 12: causal stays causal).
2. **Document context layer:** a 2-layer BiGRU (d=256) over the sentence vectors, plus a learned **domain embedding** (CL/IT).
3. **Output:** linear → 13 labels → **CRF** (learned transitions, Viterbi decoding). A fallback is softmax plus Viterbi using the train transition matrix.
4. **Loss:** CRF NLL, or class-weighted CE (sqrt inverse frequency) for the softmax/KD variants.
5. **Batching:** one document chunk per step (≤ 128 sentences, random crop, whole doc at eval). Gradient checkpointing is used for full FT.

**Context ladder** (on A + LoRMA). This answers "does context help?":
`S + linear` → `S + BiGRU` → `W + linear` → `W + BiGRU` → `W + BiGRU + CRF`.

## 3. Experiments (the required ≥ 4, plus extras)
| # | Arch | Technique | Purpose |
|---|---|---|---|
| **R1** | A d768 (140M) | **LoRMA⁺** r=8 (compulsory) | Main model; runs the context ladder |
| **R2** | A d768 | **Full fine-tuning** | PEFT vs full FT on the same backbone; **KD teacher** |
| **R3** | B encoder (~41M) | **rsLoRA** (α/√r scaling) | Native bidirectional encoder; rsLoRA vs LoRA |
| **R4** | C prefix-LM (58M) | **IA³** (learned scaling of k, v, FFN) | Very few trainable params; uses C's bidirectional prefix mode |
| **R5** | D efficient (54M) | **Prefix tuning** (per-layer K/V prefixes, always visible through the sliding window) | 2048 context, so a sentence can see more of the document |
| **R6** | A d560 (58M) student | **Knowledge distillation** from R2, or from the R7 ensemble if it's stronger | Sentence-level KL on teacher marginals (T=2) + hard CE, optionally + MSE on sentence vectors. Report params, VRAM and speed vs the teacher |
| **R7** | ensemble | Average the per-sentence label marginals of the best models → Viterbi | Phase 2's biggest reliable gain (+0.036 on test) |

Every PEFT run trains its BiGRU/CRF head fully (~2–3M params), and this is reported separately from the backbone's trainable params.

Extras, if time allows:
- a **method-controlled comparison** (LoRMA on B/C/D) to separate the architecture effect from the method effect, which was a known confound in Phase 2
- joint vs per-domain models
- rare-label handling (suppress Dissent/PrecOverruled vs keep them)
- a final retrain on train+dev with a fixed epoch count and the **same LR schedule**. Phase 2 obs. 16: the all-data retrain hurt because the schedule changed.

**For every run, record:** dev macro-F1 (both averaging variants) overall and **per domain**, micro-F1/accuracy, per-label P/R/F1, the confusion matrix, trainable params, peak VRAM, train time and inference time.

## 4. Shared pipeline (`Phase3/src/`, reusing Phase 2 code)
- Reused as-is from `Phase2/src`: `backbone.py` (load_backbone), `bpe.py`, `models_a/b/c/d.py`
- Reused and extended: `peft.py`. Already has LoRA and LoRMA. Add rsLoRA (a scaling flag), IA³ and prefix tuning (needs a K/V-prefix hook in each architecture's attention).
- New:
  - `rr_data.py`: load and tokenize, S/W packing, sentence-span index
  - `heads.py`: BiGRU, CRF
  - `train_rr.py`: loop, early stopping on dev macro-F1, KD loss
  - `evaluate.py`: macro-F1 in both variants, per domain
  - `submit.py`:
    - writes `prediction.json`
    - **validates** it: 10 IDs, `len(pred) == len(text)` for each doc, labels exactly as in `label_vocab.json`
    - zips it
- **Sanity checks before any model:**
  - gold → prediction.json → scorer = 1.0
  - majority-class and "label of previous sentence = oracle" baselines, to understand the scale of the metric

## 5. Submissions (minimum 4 valid, aim for 7–8, max 10/day)
One per experiment R1–R6, then R7 ensemble, plus at most one post-processing or all-data variant. Upload each one as soon as it exists, since Codabench evaluation takes 15–30 min. **The first upload (R1) also tells us how macro-F1 treats absent labels**, by comparing the Codabench score to the two local variants.

## 6. Schedule
| Day | Work |
|---|---|
| Sat 11 Oct | Phase 2 deadline comes first. Phase 3 prep: data downloaded ✔, stats ✔, plan ✔ |
| Sun 12 Oct | Write `src/`, run the sanity checks, run the R1 context ladder. **Upload R1** |
| Mon 13 Oct | R2 full FT, R3 B rsLoRA, R4 C IA³. Upload all three. Ask on Discord how macro-F1 handles absent labels |
| Tue 14 Oct | R5 D prefix tuning, R6 KD. Upload both. Run the method-controlled extras (LoRMA on B/C/D) |
| Wed 15 Oct | R7 ensemble; per-domain and rare-label ablations; all-data retrain. Upload. Error analysis: confusions (ArgPet↔ArgResp, PrecRelied↔PrecNotRelied, Ratio↔RulingPresent, Fact↔RulingLower), CL vs IT, position in document |
| Thu 16 Oct | Figures, `analysis/results.md`, finish `Phase 3.txt`. Final uploads **by the afternoon** |

## 7. Folder layout (same as Phase 2)
```
Capstone Project/Phase3/
├── Phase3_Plan.md            this plan
├── Phase 3.txt               running log (results, Codabench scores, observations)
├── src/  notebooks/  data/
├── submissions/<EXP>/submission_<stamp>_<RUN_ID>/   prediction.json, submission.zip, run_info.json
└── analysis/                 results.md/.csv, figures/, error_analysis/

C:\kaggle\working\cs781\phase3\{cache, runs\<EXP>\<RUN_ID>\, logs}   (large files, outside OneDrive)
```
