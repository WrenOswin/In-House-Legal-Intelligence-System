# Phase 2 – Legal NER: Experiment Plan

Window: **7–11 Oct 2026** (deadline 11 Oct, 23:59 IST). Metric: **strict macro-F1** over 12 labels. A span counts only when start, end and label all match exactly.

## 1. What the data looks like
The source is `C:\kaggle\working\cs781\hf_data\Phase_2_LNER`. The full stats are in `Phase 2.txt` §2.

| | |
|---|---|
| Train | 70 docs (fold1 35 + fold2 35), 8,393 entities (~120/doc), 0 overlapping spans |
| Test | 35 docs (`fold3`); **doc `1305957` is empty**, so it must still be in prediction.json as `[]` |
| Doc length | 10k–114k chars, median 27k → **2.2k–24k tokens** (median ~5.6k) with our BPE |
| Total train tokens | ~465k, so one epoch of a 140M model takes about a minute and experiments are cheap |

**Label counts (number of docs containing the label in brackets):** DATE 1575 (70), STAT 1260 (65), PREC 1183 (66), COURT 1013 (66), AUTH 1003 (43), CASENO 760 (65), APP 407 (63), RESP 381 (63), JUDGE 245 (66), A.COUNSEL 195 (49), WIT 187 (**12**), R.COUNSEL 184 (47).

What this means for the design:
- **Macro-F1:** WIT counts as much as DATE, but it appears in only 12 docs. Class balancing therefore matters.
- **Position:** APP, RESP, counsel and judge mentions cluster in the header (median position 0.15–0.27 of the doc). The first window carries most of this signal.
- **Long spans:** PREC can run to 195 chars or 52 tokens, e.g. full case citations. BIO tagging handles these; a span-enumeration head needs a high max width.
- **Ambiguous surface forms:** 23 strings carry more than one label. "Union of India" appears as RESP 28, APP 14, PREC 4 and AUTH 4; "Supreme Court" as COURT 114 and PREC 3. Context decides the label.
- **Unannotated repeats:** about 10% of repeated occurrences of an annotated string are left unannotated. This is a source of false positives from boilerplate.
- **Alignment:** with the Phase 1 BPE, only 1 start and 152 ends of 8,393 entities fall inside a token. Most cases are glued punctuation such as `SCC 47].` or `1063);`. A few cut inside a word, such as "Railway Board|s". **Fix:** during NER, split runs of punctuation into single characters before applying BPE. The rest (~0.3%) is the accuracy ceiling for token-level modelling.

## 2. Phase 1 models available

| Arch | Run | Params | Tokenizer | Context | Attention |
|---|---|---|---|---|---|
| A | `A_decoder_only/d768_L12_T1024_v35k_5fbdf6` | 140.2M | v35k | 1024 | causal |
| A-GQA | `A_decoder_only_gqa/d576_L23_T1024_v35k_fbb29e` | 132.1M | v35k | 1024 | causal |
| A-small | `A_decoder_only/d640_L8_T1024_v35k_42d59e` | 74.8M | v35k | 1024 | causal |
| B | `B_encoder_decoder/d512_E6D6_S768T256_v32k_05421b` | 73.0M (encoder ~41M) | v32k | 768 (enc) | bidirectional encoder |
| C | `C_prefix_lm/d560_L8_T1024_v32k_6cce41` | 58.1M | v32k | 1024 | prefix = bidirectional |
| D | `D_efficient/d560_L8_T2048_v32k_05e1ef` | 54.3M | v32k | 2048 | causal, sliding window 256 + global every 4th layer |

`model_final.pt` stores only the weights, `model_kw` and the class name. The class code lives in the Phase 1 notebooks, so it gets copied into `src/phase1_models.py`.

**A causal model is a weak NER encoder:** a token can't see the words to its right. For A and D the attention mode is an experimental factor:
1. causal, as pretrained
2. causal attention removed during fine-tuning (bidirectional)

C and B are bidirectional natively. For C, the whole window is treated as the prefix.

## 3. Shared pipeline (`Phase2/src/`, used by every notebook)
1. **`phase1_models.py`:** the four Phase 1 classes, plus `load_backbone(run_dir, attn_mode)` that returns per-token hidden states.
2. **`ner_data.py`:** an offset-aware tokenizer (char span per token, punctuation split), BIO labels (25 tags), and sliding windows of the model's context length with 25% overlap. The val split is fixed in `data/val_split.json`: 56 train / 14 val docs, each label ~20% in val, WIT in both sides.
3. **`peft.py`:** LoRA, **LoRMA** (`h = W₀x + (α/r)·B(A(W₀x))`, LoRMA⁺; LoRMA-π optional), Pfeiffer/Houlsby adapters, BitFit, prefix tuning, layer freezing. All of these wrap `nn.Linear`, so they work on all four architectures (fused `qkv`/`w12` included).
4. **`heads.py`:** a linear BIO head, a CRF head, and optionally a span head.
5. **`train_ner.py`:** the training loop (bf16, early stopping on val macro-F1), class-weighted CE or focal loss, and the KD loss.
6. **`decode_eval.py`:**
   - merge overlapping windows (keep the logits of the window where the token is most central)
   - BIO repair and Viterbi decoding → char spans
   - trim whitespace at span edges
   - **local strict macro-F1** in the same form as Codabench
   - write `prediction.json`, validate it (all 35 IDs, int offsets, valid labels) and zip it

Validate the pipeline before any model: converting gold labels to BIO and back must give F1 ≈ 0.997 on train. This checks the offsets end to end.

## 4. Experiments (the required ≥4, plus extras)
Each experiment differs by **method**, not just hyperparameters, and together they cover all four architectures.

| # | Arch | Technique | Head | Purpose |
|---|---|---|---|---|
| **E1** | A d768 | **Full fine-tuning** | CRF | Strongest baseline and **KD teacher**. Ablations: causal vs bidirectional; CE vs class weights |
| **E2** | A d768 | **LoRMA⁺** (compulsory), r=8, on qkv/proj/w12/proj | CRF | Same backbone as E1, so it shows directly whether PEFT gets close to full FT |
| **E3** | B encoder | **LoRA** r=8 | linear + CRF | Native bidirectional encoder; LoRA vs LoRMA contrast |
| **E4** | C prefix-LM | **Pfeiffer adapters** | CRF | Uses the bidirectional prefix mode C was pretrained with |
| **E5** | D efficient | **Partial / layer-wise FT** (top 4 layers + head) or BitFit | CRF | 2048-token context, so it tests the context-window factor |
| **E6** | A d640 student | **Knowledge distillation** from the E1 teacher | CRF | Same v35k tokenizer, so token-level KL on soft labels plus hard CE. Reports params, memory and speed vs the teacher |

Extras, if time allows: LoRMA-π vs LoRMA⁺, LoRMA on B/C, span head vs BIO, A-GQA (deep-thin) full FT, a final retrain of the best config on all 70 docs.

**For every run, record:** val strict macro/micro-F1, per-label P/R/F1, boundary-error vs label-error counts, trainable params, peak VRAM, train time, and inference time.

## 5. Submissions (minimum 4 valid, aim for 6–8, max 10/day)
One submission per experiment E1–E6. For the best config, add one more trained on all 70 docs (train+val). Upload early: Codabench evaluation takes 15–30 minutes.

## 6. Schedule

| Day | Work |
|---|---|
| Tue 7 Oct (tonight) | Data analysis ✔, folder layout ✔, val split ✔. Write `src/` and check the gold round-trip |
| Wed 8 Oct | E1 (with ablations), first submission. E2 LoRMA, submit |
| Thu 9 Oct | E3 (B), E4 (C), E5 (D), submit each |
| Fri 10 Oct | E6 KD, submit. Extras and the all-data retrain of the best model |
| Sat 11 Oct | Error analysis (per-label, boundary vs label confusion, long docs, boilerplate FPs), finish `Phase 2.txt`. Last uploads before the evening |

## 7. Folder layout

```
Capstone Project/Phase2/                    (OneDrive: code, small outputs, report material)
├── Phase2_Plan.md                          this plan
├── Phase 2.txt                             running log (results, scores, observations)
├── src/                                    shared pipeline modules (section 3)
├── notebooks/                              Phase2_E1_A_FullFT.ipynb ... Phase2_E6_KD.ipynb
├── data/val_split.json                     fixed train/val doc split
├── submissions/<EXP>/submission_<stamp>_<RUN_ID>/   prediction.json, submission.zip, run_info.json
└── analysis/                               metrics tables, figures/, error_analysis/

C:\kaggle\working\cs781\phase2\             (outside OneDrive: large files)
├── cache/<tokenizer>/                      tokenized windows + offsets
├── runs/<EXP>/<RUN_ID>/                    config.json, ckpt_best.pt, history.json,
│                                           val_predictions.json, val_metrics.json
└── logs/
```

`RUN_ID` = backbone tag + technique + a hash of the training config, as in Phase 1.
