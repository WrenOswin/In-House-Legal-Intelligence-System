"""
Pipeline checks before any model (python sanity.py):
  1. gold labels -> prediction.json writer/validator -> scorer must give 1.0
  2. reference baselines on dev: majority label per domain, and an oracle that copies the previous
     sentence's GOLD label (upper bound of pure label persistence, shows how much the runs explain)
  3. how the two macro-F1 variants differ on dev
"""
import tempfile
from collections import Counter

from evaluate import format_scores, rr_scores, write_submission
from rr_data import DOMAINS, LABELS, load_rr

data = load_rr()
dev = data["dev"]
gold = {d.id: [LABELS[i] for i in d.labels] for d in dev}
domain_of = {d.id: DOMAINS[d.domain] for d in dev}

# 1. round trip through the submission writer (uses the dev docs as if they were the test set)
with tempfile.TemporaryDirectory() as tmp:
    import json
    write_submission(gold, dev, tmp)
    back = json.load(open(f"{tmp}/prediction.json", encoding="utf-8"))
sc = rr_scores(back, gold, domain_of)
assert abs(sc["macro_f1_present"] - 1) < 1e-9 and abs(sc["accuracy"] - 1) < 1e-9
print(f"1. gold round trip: macro-F1 {sc['macro_f1_present']:.4f}, acc {sc['accuracy']:.4f}  OK")

# 2. baselines
maj = {dom: Counter(l for d in data["train"] if d.domain == di for l in d.labels).most_common(1)[0][0]
       for di, dom in enumerate(DOMAINS)}
pred_maj = {d.id: [LABELS[maj[DOMAINS[d.domain]]]] * len(d) for d in dev}
sc = rr_scores(pred_maj, gold, domain_of)
print(f"\n2a. majority label per domain ({ {k: LABELS[v] for k, v in maj.items()} }):")
print(format_scores(sc).split("\n")[0])
pred_prev = {d.id: [LABELS[d.labels[0]]] + [LABELS[i] for i in d.labels[:-1]] for d in dev}
sc = rr_scores(pred_prev, gold, domain_of)
print("\n2b. oracle 'previous gold label' (not a real model):")
print(format_scores(sc))
