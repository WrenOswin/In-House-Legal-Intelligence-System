# Phase 6 – Abstractive Legal Summarization: Experiment Plan

Window: **27 Oct – 2 Nov 2026** (deadline 2 Nov, 23:59). Task: generate one abstractive summary per test judgment. Primary metric: **BERTScore F1**. Secondary metrics: ROUGE-1 F1 and ROUGE-L F1. Output: `[{"id": ..., "summary": "..."}]` in `prediction.json`, zipped.

Rules (capstone doc §12.2): start from the Phase 1 models and use **all four architectures**. Run **≥ 4 substantially different experiments**: ≥ 3 different FT/PEFT techniques from the §12.2.1 list (**LoRMA compulsory**) plus ≥ 1 **knowledge distillation** run (a larger teacher distilled into a smaller student). Hyperparameter or decoding changes don't count as new experiments. Make ≥ 6 valid submissions (README), at most 10 per day.

## 1. Data (details: `Phase 6.txt` §2, `analysis/data_eda.txt`)
- Train: **7,030** Supreme Court judgments with headnote-style summaries (IN-Abs style). Test: **100** judgments. There is no dev split, so **val = 300 train docs** held out (seeded; docs that occur twice in train are never in val). Train has 6,730 docs.
- Documents: median 3,145 words (~3.7k BPE tokens), p90 7.9k, max 140k. Summaries: median 639 words (~750 tokens), p90 1.5k. A summary is ~22% of its document.
- Summaries are fairly extractive: 33% of their word bigrams are not in the document. Summary content comes from the whole judgment. The first 10% of sentences is the least used (case header, counsel names).
- **Long documents don't fit:** a 1,500-token context covers only part of the document. Coverage of reference content words: whole doc 0.90, first ~1,200 words 0.57, *selected* ~1,200 words 0.64.

## 2. Long-document strategy
- **Importance-based context selection** (`summ.Salience`) is used by all runs. A ridge regression on hashed word uni+bigram TF-IDF, position and length features predicts how much of each sentence reappears in the reference summary. It is fitted on train docs. The top sentences are kept in document order until the token budget is full, and the first sentence (case header) is always kept. As an extractive reference (no Phase-1 model), select-500 words gets ROUGE-1 0.540 vs lead-500 0.464.
- **RoPE position interpolation** (positions ÷ 2) lets A and C (pretrained at 1,024) read **2,048 tokens: 1,533 source + 512 summary**. D is natively 2,048.
- **B: Fusion-in-Decoder.** The same selected source is cut into 3 × 512-token chunks. Each chunk is encoded separately (B's encoder max is 768), and the decoder cross-attends to the concatenation. This is a hierarchical encoder–decoder representation. B's decoder positions are also interpolated ×2 (pretrained target length 256 → 512).
- **Ablation S7:** same as S2 but with the *first* 1,533 tokens (`head`) instead of selection. This measures what selection buys.

## 3. Model (`src/seq2seq.py`)
- A, C and D take `<|bos|> source <|sep|> summary <|endoftext|>`, with loss on summary tokens only. A and D are causal (as pretrained). C is a prefix-LM: the source is bidirectional and the segment embedding marks source vs summary.
- B is encoder–decoder: `<|bos|> summary` with cross-attention to the FiD memory.
- The forward passes are re-implemented on the loaded Phase-1 weights (checked: teacher-forced loss identical to the original models; cached greedy decoding identical in fp32). This gives every architecture the same extras: logits only on summary positions, batched left-padded **KV-cache generation**, repetition penalty, no-repeat-n-gram, and min/max length.
- Targets are summaries truncated to 512 tokens (≈ 440 words). BERTScore's encoder truncates at 512 sub-words anyway (≈ 400 words).

## 4. Experiments (`src/run_queue.py`)
| | Arch | Technique (§12.2.1) | lr | Purpose |
|---|---|---|---|---|
| S1 | A d768 (140M) | **LoRMA+** r=16 (compulsory) | 5e-4 | strongest Phase-1 LM; KD teacher |
| S2 | D d560 (54M) | **LoRA** r=16 | 1e-3* | native 2k window (sliding-window + global layers) |
| S3 | C d560 (58M) | **Houlsby adapters** (after attention + FFN, norms trained), b=128 | 1e-3* | prefix-LM: bidirectional source |
| S4 | B d512 (42M enc-dec) | **Pfeiffer adapters** b=128 + FiD | 1e-3* | encoder–decoder, hierarchical input |
| S5 | A d560 (58M) | **Knowledge distillation**, sequence-level (Kim & Rush): student trained on S1's summaries of the train docs (50%) + references (50%), full FT | 1e-4* | mandatory KD |
| S6 (opt.) | A d560 | full FT on references only | 1e-4* | separates KD from full FT (as in Phases 4–5) |
| S7 (opt.) | D d560 | LoRA, `head` context | 1e-3* | long-document strategy ablation |

\* lr set by 800-doc pilots (`pilot_lr` runs). Token-level KD is impossible here: the 140M teacher uses a 35k BPE vocabulary and the 58M students use 32k, so the student learns from teacher-generated summaries.

S1–S5 cover all four architectures and 5 listed techniques (LoRMA, LoRA, Houlsby, Pfeiffer, KD), which meets the requirements. Submissions: S1–S7 (≥ 6 distinct methods), plus decoding variants of the best run (extra, not counted as new methods).

**Decoding study (`src/decode_sweep.py`, no training):** max_new 256/384/512, min_new, no-repeat 0/4/6, repetition penalty 1.0/1.1/1.2 on 100 val docs. It reports ROUGE, length and 4-gram repetition, for the "generation length vs metrics" and "redundancy" analysis.

## 5. Cost (measured in pilots, duty 1.0; queue runs at duty 0.6)
- A d768 LoRMA: 20.9k tok/s, 4.2 GB peak. One epoch is ~13.8M tokens → ~18 min at duty 0.6, so 3 epochs plus evals take ~1.1 h.
- d560 models are ~1.4× faster.
- Teacher generation of 6,730 train summaries: 32 docs per ~20 s at 512 tokens → ~1.2 h at duty 1, ~2 h throttled.
- Total ≈ 8–9 h. Order: S1 → teacher generation → S5 → S2 → S3 → S4 → S6 → S7.

## 6. Risks and checks
- **No local BERTScore.** It needs a pretrained RoBERTa, so val selection uses ROUGE, length and repetition. Codabench scores will show how BERTScore follows length.
- **Degenerate repetition** in greedy decoding: no-repeat-4-gram on by default, rep4 is reported.
- **Laptop safety:** duty 0.6, vram_fraction 0.75, below-normal priority, peak VRAM measured in pilots (≤ 4.3 GB).
