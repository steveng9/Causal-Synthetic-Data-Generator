"""Sweep the Small Causal Graphs (graphs.json; python run.py && python plot.py): each graph x cut rule x cut strength x seed.

World = the graph with every edge intact. Release = the same graph sampled with the rule's
edges cut on that share of rows (the ideal DECAF: true mechanisms, cut at generation time).
A recipient trains logistic regression on the release and is scored on a fresh real table.
"""
import json, os, sys, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import csdg

GRAPHS = os.path.join(HERE, 'graphs.json')
OUT = os.path.join(HERE, 'results.json')
ROWS, SEEDS, STRENGTHS, RULES = 20000, range(1, 6), [0.0, 0.25, 0.5, 0.75, 1.0], ['ftu', 'dp', 'cf']
EDGE = 0.25          # variance share each edge explains, before the 0.9 cap on a child's total


def load_sections():
    return [(s['title'], s['sub'], [(g['title'], g['note'], g['nodes'], g['edges']) for g in s['graphs']]) for s in json.load(open(GRAPHS))]


def role(l):
    return 'hidden' if l.endswith('~') else {'S': 'protected', 'Y': 'outcome', 'A': 'admissible'}.get(l[0], 'other')


def build(ns, es):
    nodes, ids = [], {}
    for i, tok in enumerate(ns.split(','), 1):
        l, x, y = tok.split()
        r, name = role(l), l.replace('~', '')
        ids[name] = i
        nd = dict(id=i, name=name, role=r, gx=float(x), gy=float(y))
        nd.update(type='cat', card=2, shape='uniform') if r in ('protected', 'outcome') else nd.update(type='cont', cshape='normal', lo=0, hi=1)
        nodes.append(nd)
    E = [e.split('>') for e in es.split()]
    indeg = {}
    for a, b in E:
        indeg[b] = indeg.get(b, 0) + 1
    edges = [dict(id=k, a=ids[a], b=ids[b], kind='dir', sign=1, strength=min(EDGE, 0.9 / indeg[b])) for k, (a, b) in enumerate(E)]
    return dict(nodes=nodes, edges=edges)


def visible_cuts(g, rule):
    hid = {n['id'] for n in g['nodes'] if n['role'] == 'hidden'}
    v = dict(nodes=[n for n in g['nodes'] if n['id'] not in hid],
             edges=[e for e in g['edges'] if e['a'] not in hid and e['b'] not in hid])
    return csdg.fair_cuts(csdg.normalize(v), rule)


def lr():
    return LogisticRegression(max_iter=500)


def metrics(g, syn, real):
    names = {n['role']: [] for n in g['nodes']}
    for n in g['nodes']:
        names[n['role']].append(n['name'])
    S, Y = names['protected'], names['outcome'][0]
    feats = [n['name'] for n in g['nodes'] if n['role'] != 'hidden' and n['name'] != Y]
    Xs = np.column_stack([syn[c] for c in feats]).astype(float)
    Xr = np.column_stack([real[c] for c in feats]).astype(float)
    m = lr().fit(Xs, syn[Y])
    p = m.predict_proba(Xr)[:, 1]
    gap = max(abs(p[real[s] == 1].mean() - p[real[s] == 0].mean()) for s in S) * 100
    util = skill(real[Y], p)
    inf = []
    for s in S:
        fa = [c for c in feats + [Y] if c not in S]
        if not fa:
            inf.append(0.0)
            continue
        a = lr().fit(np.column_stack([syn[c] for c in fa]).astype(float), syn[s])
        inf.append(skill(real[s], a.predict_proba(np.column_stack([real[c] for c in fa]).astype(float))[:, 1]))
    return gap, max(inf), util


def skill(y, p):
    """Brier skill on real rows: 1 - Brier / Brier of always predicting the real base rate."""
    b = y.mean()
    return 1 - np.mean((p - y) ** 2) / (b * (1 - b))


def main():
    secs = load_sections()
    assert sum(len(c) for _, _, c in secs) == 46, [len(c) for _, _, c in secs]
    out, k, t0 = [], 0, time.time()
    for title, sub, cards in secs:
        sec = dict(title=title, sub=sub, graphs=[])
        for t, note, ns, es in cards:
            k += 1
            g = build(ns, es)
            cuts = {r: sorted(visible_cuts(g, r)) for r in RULES}
            reals = {s: csdg.sample(g, rows=ROWS, seed=1000 + s)[1] for s in SEEDS}
            res = {}
            for r in RULES:
                rows = []
                for st in STRENGTHS:
                    vals = []
                    for s in SEEDS:
                        syn = csdg.sample(g, rows=ROWS, seed=s, cut=set(cuts[r]) if st > 0 else None, cut_strength=st if st > 0 else 1.0)[1]
                        vals.append(metrics(g, syn, reals[s]))
                    v = np.array(vals)
                    rows.append(dict(mean=v.mean(0).round(4).tolist(), sd=v.std(0, ddof=1).round(4).tolist()))
                res[r] = rows
            sec['graphs'].append(dict(k=k, t=t, note=note, nodes=g['nodes'],
                                      edges=[dict(id=e['id'], a=e['a'], b=e['b'], strength=round(e['strength'], 4)) for e in g['edges']],
                                      cuts=cuts, res=res))
            print(f'{k:2d} {t[:34]:34s} cuts {cuts}  {time.time() - t0:5.0f} s', flush=True)
        out.append(sec)
    json.dump(dict(rows=ROWS, seeds=list(SEEDS), strengths=STRENGTHS, edge=EDGE, metrics=['score_gap_pp', 's_inference_brier_skill', 'utility_brier_skill'], sections=out), open(OUT, 'w'))


if __name__ == '__main__':
    main()
