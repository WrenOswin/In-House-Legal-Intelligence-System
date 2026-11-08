"""
Document-level post-processing of predicted spans.

propagate_strings: in the gold data every standalone repeat of an annotated string is annotated with the
same label (checked on the 56 training docs: 100% for all 12 labels, 0 exceptions), so within a document
every occurrence of a predicted string gets the same (majority) label, and missed occurrences are added.

resolve_party_roles: party-role resolution for APP / RESP (no new entities are added).

The role of a party is written once, in the case title: "<party> ... Appellant/Petitioner ... Versus ...
<party> ... Respondent". The model finds party names but often puts later mentions on the wrong side.

1. Header: every predicted APP/RESP span in the case title is set by its side of "Versus" / "Vs.".
2. Body: every later APP/RESP prediction whose name matches header parties of exactly one side gets that side.
"""
import re

VERSUS = re.compile(r"(?i)\b(?:versus|vs\.?|v/s)(?![a-z])")
HEADER_CHARS = 2500          # case titles sit at the start of the document
SIDE_CHARS = 300             # how far a party name may be from "Versus"
STOP = {"the", "of", "and", "ors", "anr", "others", "another", "ltd", "limited", "pvt", "private", "co", "m/s",
        "state", "india", "union", "government", "govt", "shri", "smt", "mr", "mrs", "dr", "through", "its",
        "company", "corporation", "&", "petitioner", "petitioners", "respondent", "respondents", "appellant",
        "appellants", "no", "nos", "s", "and/or"}


def _words(s):
    return {w for w in re.findall(r"[a-z0-9/&]+", s.lower()) if w not in STOP and len(w) >= 3}


RESP_LABEL = re.compile(r"(?i)^[\s.:,-]*respondents?\s*(?:\(s\))?\s*:")   # "Vs. RESPONDENT: <name>" format
RIGHT_END = re.compile(r"(?i)respondent|date of judgment|j\s?u\s?d\s?g\s?m\s?e\s?n\s?t|\border\b|\bwith\b|through")


def header_sides(text):
    """[(left_start, versus_start, versus_end, right_end)] for every 'versus' in the case title.
    The respondent side ends at the first header marker after the respondent's name ('Respondent(s)',
    'Date of judgment', 'Judgment', 'With', 'Through'), so it never reaches into the body or the next case."""
    out, prev_end = [], 0
    head = text[:HEADER_CHARS]
    for m in VERSUS.finditer(head):
        left = max(prev_end, m.start() - SIDE_CHARS)
        start = m.end()
        lab = RESP_LABEL.match(head, start)
        if lab:
            start = lab.end()
        nxt = VERSUS.search(head, start)
        cap = min(m.end() + SIDE_CHARS, nxt.start() if nxt else len(head))
        stop = RIGHT_END.search(head, start, cap)
        right = stop.start() if stop else cap
        out.append((left, m.start(), m.end(), right))
        prev_end = right
    return out


def resolve_party_roles(text, ents):
    """ents: [[s, e, label], ...] -> same spans, APP/RESP labels corrected."""
    sides = header_sides(text)
    if not sides:
        return ents
    ents = [list(x) for x in ents]
    names = {"APP": set(), "RESP": set()}
    in_header = set()
    for i, (s, e, l) in enumerate(ents):
        if l not in ("APP", "RESP"):
            continue
        for left, vs, ve, right in sides:
            if left <= s and e <= vs:
                ents[i][2] = "APP"; break
            if ve <= s and e <= right:
                ents[i][2] = "RESP"; break
        else:
            continue
        in_header.add(i)
        names[ents[i][2]].add(text[s:e])
    words = {k: set().union(*(_words(n) for n in v)) if v else set() for k, v in names.items()}
    full = {k: {" ".join(n.lower().split()) for n in v} for k, v in names.items()}
    for i, (s, e, l) in enumerate(ents):
        if l not in ("APP", "RESP") or i in in_header:
            continue
        name = " ".join(text[s:e].lower().split())
        hit = {k for k in names if name in full[k] or any(name in f for f in full[k])}
        if not hit:
            w = _words(name)
            hit = {k for k in names if w and w & words[k]}
        if len(hit) == 1:
            ents[i][2] = hit.pop()
    return ents


NAME_LABELS = ("APP", "RESP", "A.COUNSEL", "R.COUNSEL", "JUDGE", "WIT")   # people and parties


def propagate_strings(text, ents, min_len=4, labels=None):
    """Make every standalone occurrence of a predicted string carry that string's majority label.
    Occurrences overlapping another (kept) prediction are skipped, longer strings are placed first.
    labels: only strings whose majority label is in `labels` are propagated (None = all). Names of people and
    parties propagate well; generic strings ('High Court', 'Act') often sit inside a larger entity the model
    did not predict, so propagating them adds false positives (val: COURT fp 49 -> 105)."""
    from collections import Counter, defaultdict
    votes = defaultdict(Counter)
    for s, e, l in ents:
        votes[text[s:e]][l] += 1
    label_of = {name: c.most_common(1)[0][0] for name, c in votes.items()}
    out = [[s, e, label_of[text[s:e]]] for s, e, _ in ents]                 # unify labels
    taken = sorted((s, e) for s, e, _ in out)
    starts = [s for s, _ in taken]

    import bisect
    def free(s, e):
        i = bisect.bisect_left(starts, e)
        return not any(ts < e and s < te for ts, te in taken[max(0, i - 50):i])

    for name in sorted(label_of, key=len, reverse=True):
        if len(name) < min_len or (labels is not None and label_of[name] not in labels):
            continue
        for m in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text):
            s, e = m.span()
            if free(s, e):
                out.append([s, e, label_of[name]])
                i = bisect.bisect_left(starts, s)
                starts.insert(i, s); taken.insert(i, (s, e))
    return sorted(out)
