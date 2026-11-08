import json, hashlib, re, collections, statistics as st, sys
D = r'C:/kaggle/working/cs781/hf_data/Phase_5_LSI/'
tr = json.load(open(D + 'Train/train.json', encoding='utf8'))
dv = json.load(open(D + 'Train/dev.json', encoding='utf8'))
te = json.load(open(D + 'Test/test.json', encoding='utf8'))

def q(xs, ps=(0.1, 0.5, 0.9, 0.99)):
    xs = sorted(xs); return {p: xs[min(len(xs) - 1, int(p * len(xs)))] for p in ps} | {'max': xs[-1], 'mean': round(sum(xs) / len(xs), 1)}

for name, d in [('train', tr), ('dev', dv), ('test', te)]:
    print(f'== {name}: {len(d)} docs, keys {collections.Counter(tuple(sorted(x)) for x in d).most_common(3)}')
    print('  unique ids', len({x["id"] for x in d}), ' empty text', sum(not x['text'].strip() for x in d))
    print('  words', q([len(x['text'].split()) for x in d]))
    if 'labels' in d[0]:
        nl = [len(x['labels']) for x in d]
        print('  labels/doc', q(nl), ' zero-label docs', nl.count(0), ' dup labels in doc', sum(len(set(x['labels'])) != len(x['labels']) for x in d))
        print('  label types', collections.Counter(type(l).__name__ for x in d for l in x['labels']))
        print('  labels/doc dist', sorted(collections.Counter(nl).items())[:12])
    print('  sample text:', repr(d[0]['text'][:300]))
    print('  <ENTITY> per doc mean', round(sum(x['text'].count('<ENTITY>') for x in d) / len(d), 1),
          ' lowercase-only docs', sum(x['text'] == x['text'].lower() for x in d))

ct, cd = collections.Counter(l for x in tr for l in x['labels']), collections.Counter(l for x in dv for l in x['labels'])
labs = sorted(set(ct) | set(cd))
print('\n== labels: train', len(ct), 'dev', len(cd), 'union', len(labs), 'range', min(labs), max(labs))
print('  in dev not train', sorted(set(cd) - set(ct)), ' in train not dev', sorted(set(ct) - set(cd))[:30], len(set(ct) - set(cd)))
print('  missing ids in range', sorted(set(range(min(labs), max(labs) + 1)) - set(labs))[:40])
fr = sorted(ct.values(), reverse=True)
print('  train freq top20', ct.most_common(20))
print('  train freq bottom20', ct.most_common()[-20:])
for k in (1, 2, 5, 10, 20, 50, 100, 500):
    print(f'  labels with <= {k} train docs: {sum(v <= k for v in ct.values())}')
tot = sum(fr); print('  share of label mass in top 10 / top 20:', round(sum(fr[:10]) / tot, 3), round(sum(fr[:20]) / tot, 3))
print('  dev freq per label (sorted by train freq):', [(l, ct[l], cd.get(l, 0)) for l, _ in ct.most_common()])
# co-occurrence
pairs = collections.Counter(p for x in tr for p in __import__('itertools').combinations(sorted(set(x['labels'])), 2))
print('  top co-occurring pairs', pairs.most_common(15))
sets = collections.Counter(tuple(sorted(set(x['labels']))) for x in tr)
print('  distinct label sets', len(sets), ' top sets', sets.most_common(10))

# overlap / leak
def norm(t): return re.sub(r'\s+', ' ', t.lower()).strip()
def h(t): return hashlib.md5(norm(t).encode()).hexdigest()
H = {n: collections.defaultdict(list) for n in ('train', 'dev', 'test')}
for n, d in [('train', tr), ('dev', dv), ('test', te)]:
    for x in d: H[n][h(x['text'])].append(x['id'])
I = {n: {x['id'] for x in d} for n, d in [('train', tr), ('dev', dv), ('test', te)]}
print('\n== overlap')
for a, b in [('train', 'dev'), ('train', 'test'), ('dev', 'test')]:
    print(f'  {a}-{b}: shared ids {len(I[a] & I[b])}, identical (normalised) texts {len(set(H[a]) & set(H[b]))}')
for n in H: print(f'  dup texts within {n}:', sum(len(v) - 1 for v in H[n].values()))
# label consistency of exact duplicates train-dev
lt = {h(x['text']): set(x['labels']) for x in tr}
dd = [x for x in dv if h(x['text']) in lt]
print('  train-dev exact dups with same labels', sum(set(x['labels']) == lt[h(x['text'])] for x in dd), '/', len(dd))
# near-duplicate check: first 300 normalised chars + last 300
def fp(t): t = norm(t); return t[:300] + '|' + t[-300:]
F = {n: {fp(x['text']) for x in d} for n, d in [('train', tr), ('dev', dv), ('test', te)]}
for a, b in [('train', 'test'), ('dev', 'test'), ('train', 'dev')]:
    print(f'  {a}-{b}: same first+last 300 chars {len(F[a] & F[b])}')
# shingle near-dups on a sample of test
import random; random.seed(0)
def sh(t): w = norm(t).split(); return {' '.join(w[i:i + 8]) for i in range(0, max(1, len(w) - 8), 4)}
idx = collections.defaultdict(set)
pool = tr + dv
for j, x in enumerate(pool):
    w = norm(x['text']).split()
    for i in range(0, max(1, len(w) - 8), 50): idx[' '.join(w[i:i + 8])].add(j)
samp = random.sample(te, 1000); hi = 0
for x in samp:
    s = sh(x['text']); c = collections.Counter(j for g in s for j in idx.get(g, ()))
    if c:
        j, _ = c.most_common(1)[0]; s2 = sh(pool[j]['text'])
        if len(s & s2) / max(1, min(len(s), len(s2))) > 0.8: hi += 1
print('  test sample near-dups (>80% 8-gram containment) in train+dev:', hi, '/ 1000')
# does the text name the statute explicitly? e.g. 'section 302' mentions
sec = collections.Counter()
for x in tr[:2000]: sec.update(set(re.findall(r'[Ss]ection\s+(\d+[A-Z]?)', x['text'])))
print('\n== statute mentions: docs (of 2000) mentioning "Section N", top', sec.most_common(15))
print('  docs with any "section" mention', sum(bool(re.search(r'\bsection\b', x['text'], re.I)) for x in tr[:2000]), '/ 2000')
print('  docs with "<STATUTE>"-style masks:', collections.Counter(m for x in tr[:2000] for m in set(re.findall(r'<[A-Z_]+>', x['text']))).most_common(10))
