# CS781 Capstone: In-House Legal Intelligence System

**Wren Oswin, 261110073, IIT Kanpur.** Codabench username `w_261110073`.

This repository holds the code, experiment logs, analysis and submissions for the CS781 (LLMs) capstone.
A law firm cannot send privileged documents to third-party LLM APIs, so the whole system is self-hosted.
Four Transformer language models were pretrained **from scratch** on the official Indian legal corpus (Phase 1).
Every later task model (Phases 2–7) is fine-tuned from those checkpoints, using adaptation techniques written
from scratch: LoRMA, LoRA, rsLoRA, IA³, prefix tuning, Houlsby and Pfeiffer adapters, BitFit, partial and full
fine-tuning, plus knowledge distillation.
Everything was trained on one Windows laptop with an RTX 5070 Ti Laptop GPU (12 GB).

The final report is in [`Report/`](Report/) (`Report.tex` / `Report.pdf`).

## Results

Each phase has its own README. It documents the architecture, methodology and implementation, and
describes **every Codabench submission** with its local validation score and its test score.

| Phase | Task | Metric | Best Codabench score | Best submission | Submissions | Details |
|---|---|---|---|---|---|---|
| 1 | Legal LM pretraining (continue a 512-word prompt with 128 words) | BERTScore F1 | **0.8321** | A decoder-only, 140M params, 600M tokens | 6 | [Phase1/README.md](Phase1/README.md) |
| 2 | Legal NER (12 entity types, character spans) | strict macro-F1 | **0.5590** | ensemble of 3 A d768 models + party-role rules + name propagation | 11 | [Phase2/README.md](Phase2/README.md) |
| 3 | Rhetorical role prediction (13 roles per sentence) | macro-F1 | **0.4950** | D + prefix tuning, BiGRU + CRF head | 22 | [Phase3/README.md](Phase3/README.md) |
| 4 | Court judgment prediction (ILDC, binary) | macro-F1 | **0.7771** | ensemble of A d560 KD student and C full FT + KD | 11 | [Phase4/README.md](Phase4/README.md) |
| 5 | Legal statute identification (100 statutes, multi-label) | macro-F1 | **0.4035** | ensemble of four runs (best single run: KD student + per-statute attention, 0.4033) | 9 | [Phase5/README.md](Phase5/README.md) |
| 6 | Abstractive summarization | BERTScore F1 | **0.8424** | A d768 + LoRMA | 7 | [Phase6/README.md](Phase6/README.md) |
| 7 | Prior case retrieval (rank 7 candidates per query) | micro-F1@7 | **0.4312** | A d768 + LoRMA bi-encoder fused with trigram BM25 | 6 | [Phase7/README.md](Phase7/README.md) |

There are 72 valid Codabench submissions in total.

## Repository layout

```
README.md                 this file
requirements.txt          Python packages
Phase1/                   pretraining: one notebook per architecture, the phase log, submissions
Phase2/ ... Phase7/       one folder per downstream task:
    README.md             documentation of the phase and every submission
    "Phase N.txt"         full experiment log (data analysis, every run, observations, in order)
    PhaseN_Plan.md        experiment plan written before training (Phases 2, 3, 5, 6)
    src/                  all code for the phase (models, PEFT, training, decoding, scoring)
    notebooks/            Phase 2 only: one notebook per experiment (thin wrappers around src/)
    analysis/             EDA scripts and output, results tables, error analysis, figures
    data/                 small split files (e.g. the Phase-2 validation split)
    submissions/<EXP>/submission_<timestamp>_<run id>/
                          submission.zip (as uploaded), prediction.json, run_info/info/gen_config.json
Report/                   LaTeX report, bibliography, figures and the figure script
code/                     the submitted code folder: only the best model of each phase
    phase1/ ... phase7/   src/ (code of the best run), configs/ (run_info.json of the best run and
                          of its ensemble members and KD teachers), README.md (how to reproduce it)
```

Large files are **not** in the repository: datasets, tokenized caches, model checkpoints (`model_final.pt`)
and teacher outputs. The code keeps them in a work directory outside OneDrive, `C:/kaggle/working/cs781/`:

```
C:/kaggle/working/cs781/
    hf_data/Phase_<n>_<task>/      official dataset files (Hugging Face, gated)
    artifacts/tokenizers/          BPE tokenizers (bpe_v32000_*.json, bpe_v35000_*.json)
    runs/<arch>/<run id>/          Phase-1 checkpoints (model_final.pt) used by every later phase
    phase<N>/{cache,runs,logs}/    per-phase token caches, fine-tuned runs and logs
```

## Shared design

### Phase-1 backbones

All four architectures share LLaMA-style components: pre-norm RMSNorm, rotary position embeddings (RoPE),
SwiGLU feed-forward layers, no bias terms, and tied input/output embeddings.
Their tokenizers are byte-pair encoding (BPE), trained by our own code.

| Name used in later phases | Architecture | Size | Params | Pretraining |
|---|---|---|---|---|
| `A_d768` | decoder-only, 12 layers, d=768, 12 heads, 1,024 context, 35k BPE | 140.15M | 600M tokens | final Phase-1 model |
| `A_d560` | decoder-only, 8 layers, d=560, 1,024 context, 32k BPE | 58.1M | 200M tokens | architecture study; KD student |
| `B_d512` | encoder-decoder, 6+6 layers, d=512, 768-token prefix / 256-token continuation | 73.0M (encoder 41.7M) | 200M tokens | architecture study |
| `C_d560` | prefix-LM: bidirectional prefix, causal continuation, segment embedding | 58.1M | 200M tokens | architecture study |
| `D_d560` | efficient: 256-token sliding window + full attention every 4th layer, GQA (2 K/V heads), 2,048 context | 54.3M | 200M tokens | architecture study |

The data mix was 35% Acts, 35% Supreme Court, 15% High Court and 15% Tribunal text.
The course requires every downstream phase to use all four architectures, so every phase has at least one
run per architecture. Classification and retrieval tasks use B's encoder only.

### Adaptation techniques, all written from scratch

They are implemented in `PhaseN/src/peft.py`. This is our own file, despite the shared name with the
Hugging Face library. Neither HF `peft` nor `transformers` is installed or imported.

- **Full fine-tuning**, and **partial fine-tuning** (the top 4 of 8 blocks plus the final norm).
- **LoRA**: adds `(alpha/r)·B·A·x` to a frozen `W0·x`, with B initialised to zero.
  **rsLoRA** uses the same update scaled by `alpha/sqrt(r)`.
- **LoRMA** (Bihany, Patel & Modi, 2025). A multiplicative low-rank update in its identity-inflated form:
  `h = (I + (alpha/r)·B·A)·W0·x = W0·x + (alpha/r)·B(A(W0·x))`.
  It is evaluated right to left at O(d·r) cost per token and starts exactly at the pretrained model.
  It is applied to the attention and SwiGLU projections.
- **IA³**: learned rescaling of keys, values and FFN activations.
- **Prefix tuning**: per-layer key/value prefixes produced by an MLP. They also work inside D's blocked
  sliding-window kernel.
- **Houlsby** and **Pfeiffer adapters**: bottleneck MLPs.
- **BitFit**: zero-initialised biases are added, because the backbones have none, and only those are trained.
- **Knowledge distillation (KD)**. The teacher is a 140M `A_d768` and the student a 58M `A_d560`.
  They use different tokenizers (35k vs 32k), so distillation works on outputs both models share:
  - character-aligned soft tags for NER;
  - sentence- or document-level soft labels for RR, CJPE and LSI;
  - sequence-level targets for summarization;
  - the teacher's in-batch score distribution for retrieval.

  From Phase 3 on, every KD run has a **no-teacher control**: the same student and recipe without a teacher.

## Environment and how to run

- **Hardware:** Windows 11, RTX 5070 Ti Laptop GPU (12 GB).
- **Software:** Python 3.12, PyTorch 2.14 (CUDA 13.2 build). Install the packages with
  `pip install -r requirements.txt`.
- **Dataset:** [`Exploration-Lab/CS-781-Capstone`](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone)
  is gated. Request access, log in with `huggingface-cli login`, and download it into
  `C:/kaggle/working/cs781/hf_data/`. The Phase-1 notebooks do this with `snapshot_download`.
- **Phase 1:** run a notebook in `Phase1/` top to bottom. It trains the BPE tokenizer, builds the token
  corpus, pretrains with automatic checkpoint resume, and writes a submission.
- **Phases 2–7:** run from the phase's `src/` folder. Every run is identified by a hash of its
  training-relevant config, and finished runs are skipped.
  ```bash
  cd Phase5/src
  python build_cache.py                  # tokenise the splits once (Phases 5-6; other phases cache on first use)
  python run_queue.py                    # every experiment of the phase, one at a time
  python run.py '{"exp": "K1_A_lorma", "backbone": "A_d768", "method": "lorma"}'   # or a single run
  python ensemble.py --all-subsets       # ensembles of finished runs (Phases 2-5)
  ```
- **Paths:** the Phase-1 checkpoint directory and the data paths are constants near the top of
  `backbone.py` and of the phase's data module (`ner_data.py`, `rr_data.py`, `cjpe.py`, `lsi.py`, `summ.py`,
  `pcr.py`). Change them if your work directory is somewhere else.
- **Laptop throttling:** the training drivers can throttle themselves (GPU duty cycle, below-normal priority,
  a VRAM fraction) so the laptop stays usable. These settings are not part of the run ID and do not change
  the results.

## Data-integrity notes

- **Phase 4.** The released CJPE `test.json` carried labels, and the original `dev.json` contained every
  test document with its label. All labels were dropped on load. Validation used only the 994 dev documents
  outside the test set, and the leak was reported to the TAs.
- **Phase 7.** Every query is also a document in its own candidate pool, so self-retrieval is always masked.
- No external data and no pretrained weights were used. All models start from our own Phase-1 checkpoints.

## LLM usage

AI assistance (Claude) was used in this project, as declared in the report's LLM Usage Declaration.
Each phase log records what was AI-assisted.
