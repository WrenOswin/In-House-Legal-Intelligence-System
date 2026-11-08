# Phase 2 best model: ensemble of three A d768 taggers + post-processing (E7)

**Codabench strict macro-F1 0.5590** (submission `E7_ensemble/submission_20261009_204219_ens3_roles_propnames`).
Val macro-F1 0.5891. The best single model, E2 LoRMA with word snapping, scores 0.5174.

## Members

All three share the Phase-1 `A_d768` backbone and its 35k tokenizer, so their per-token probabilities can be
averaged. Each one is a backbone with a linear BIO head of 25 tags.

| Member | Technique | Attention | Trainable | Val macro-F1 | Config |
|---|---|---|---|---|---|
| E2 `A_d768_lorma_684310` | **LoRMA** r=8, α=16 on qkv, proj and SwiGLU; lr 5e-4, ≤25 epochs | causal | 1.9M | 0.5074 | `configs/E2_A_lorma_member.json` |
| E1 `A_d768_full_8fcd22` | full fine-tuning; lr 5e-5, ≤20 epochs | bidirectional | 140.2M | 0.4728 | `configs/E1_A_full_member_8fcd22.json` |
| E1 `A_d768_full_5c4900` | full fine-tuning; lr 5e-5, ≤20 epochs | causal | 140.2M | 0.5048 | `configs/E1_A_full_member_5c4900.json` |

## Pipeline

1. **Tokenisation (`ner_data.py`, `bpe.py`).** The text is never modified. Tokens carry character offsets. Gold
   spans become BIO tags on tokens, and punctuation is split into single characters so that span ends align
   with tokens. The gold → BIO → span round trip scores 0.9944.
2. **Windows.** Each document is processed in windows of the backbone's context, overlapping by half.
3. **Ensemble (`ensemble.py`).** The per-token tag probabilities of the three members are averaged.
4. **Decoding (`ner_data.py`, `postprocess.py`).** A BIO-constrained Viterbi pass gives valid spans. Word
   snapping then extends any span that cuts a word to the whole word.
5. **Post-processing (`postprocess.py`).** Both rules were designed on training documents only.
   - `resolve_party_roles`: a party's side in the case title ("… Versus …") fixes the `APP`/`RESP` label of
     its later mentions.
   - `propagate_strings`: every standalone repeat of a predicted name gets its majority label. This applies
     to `APP`, `RESP`, `A.COUNSEL`, `R.COUNSEL`, `JUDGE` and `WIT`.

## Files

| File | Role |
|---|---|
| `src/ner_data.py`, `data/val_split.json` | loading, offsets, BIO conversion, the 56/14 train/val split |
| `src/backbone.py`, `src/models_a..d.py`, `src/bpe.py` | Phase-1 checkpoint and tokenizer loading |
| `src/peft.py` | LoRMA (and the other PEFT methods) |
| `src/train_ner.py` | training loop, early stopping, Viterbi decoding, submission writer |
| `src/ensemble.py` | probability averaging and the ensemble submission |
| `src/postprocess.py` | word snapping, party roles, name propagation |
| `src/evaluate.py` | strict span-level macro- and micro-F1 |
| `notebooks/` | the three member runs (E1, E2) and the ensemble (E7) |

## Reproduce

Run the notebooks from `notebooks/` (they add `../src` to the path):

1. `Phase2_E1_A_full.ipynb`: both E1 members (bidirectional and causal).
2. `Phase2_E2_A_lorma.ipynb`: the E2 LoRMA member (the first cell; the second is the all-data retrain).
3. `Phase2_E7_ensemble.ipynb`: scores the ensemble on val. Uncomment the line
   `ensemble_submission(MEMBERS, "ens3_roles_propnames", roles=True, propagate_names=True)` to write the
   submission.
