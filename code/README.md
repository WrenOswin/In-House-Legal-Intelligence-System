# Code: best model of each phase

Each `phaseN/` folder holds only the code and configuration needed to reproduce that phase's **best
Codabench submission**. The full experiment history, including the non-best runs, analysis scripts and every
submission, lives in the `PhaseN/` folders of the repository.

| Phase | Task | Best model | Codabench |
|---|---|---|---|
| 1 | Legal LM pretraining | Decoder-only A d768 (140.15M), 600M tokens | BERTScore **0.8321** |
| 2 | Legal NER | Ensemble of 3 × A d768 (LoRMA + 2 full FT) + party roles + name propagation | macro-F1 **0.5590** |
| 3 | Rhetorical roles | D d560 (efficient) + prefix tuning + BiGRU + CRF | macro-F1 **0.4950** |
| 4 | Judgment prediction (CJPE) | Ensemble of A d560 KD student (J5) + C d560 full FT + KD (K3) | macro-F1 **0.7771** |
| 5 | Statute identification (LSI) | Mean-probability ensemble K2 + K5 + K6 + K7 | macro-F1 **0.4035** |
| 6 | Summarisation | A d768 + LoRMA r=16 (S1) | BERTScore **0.8424** |
| 7 | Prior case retrieval | A d768 + LoRMA bi-encoder fused with trigram BM25 (P1) | F1@7 **0.4312** |

## Layout

```
phase1/            Phase1_A_DecoderOnly.ipynb (tokenizer, data builder, model, training, generation)
                   gen_config_best.json (decoding settings of the best submission)
phase2..phase7/
  src/             the source files the best run imports (all from scratch, see below)
  configs/         run_info.json of the best submission and of every run it depends on
                   (ensemble members, KD teachers): full hyperparameters, params, dev scores
  README.md        what the best model is and the exact commands that reproduce it
requirements.txt
```

## Dependencies between phases

Phases 2–7 fine-tune **Phase-1 checkpoints**. `backbone.py` (or `seq2seq.py` in Phase 6) loads them from the
Phase-1 run directories together with their BPE tokenizers. The four architectures share one code base:
`models_a.py` (A, decoder-only), `models_b.py` (B, encoder-decoder), `models_c.py` (C, prefix-LM) and
`models_d.py` (D, efficient sliding-window + GQA). `backbone.py` imports all four, so every phase folder
contains them even when the best model uses only one. `peft.py` implements LoRMA, LoRA, rsLoRA, IA³, adapters,
prefix tuning, BitFit and partial fine-tuning.

## Running

- Python 3.12. Install a CUDA build of PyTorch first, then `pip install -r requirements.txt`.
- The data is the gated Hugging Face dataset of the course. Run `huggingface-cli login` before the first run.
- Paths point to the original work directory, `C:/kaggle/working/cs781`. Edit them for your machine:
  - `DATA_DIR` (`HF` in Phase 6) and `WORK` in the data module of each phase: `ner_data.py` / `train_ner.py`,
    `rr_data.py`, `cjpe.py`, `lsi.py`, `summ.py` and `pcr.py`;
  - `PHASE1_RUNS` in `backbone.py` (`seq2seq.py` in Phase 6);
  - `WORK_RUNS` in `run_queue.py`.
- Run every command from the phase's `src/` folder. Submissions are written to `phaseN/submissions/`.
- Model weights are not included because of their size. Train Phase 1 first, then the phase you want.
