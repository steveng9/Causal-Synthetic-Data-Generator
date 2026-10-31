#!/usr/bin/env python3
"""Headless sampler for the Causal Synthetic Data Generator (index.html).

Same graph + rows + seed + cut gives the same table as the page's "Download CSV".
Needs only numpy (pandas optional, for to_dataframe).

  python csdg.py list                                   # built-in examples
  python csdg.py sample sc_pbmc --rows 5000 --seeds 1-20 --out runs/pbmc
  python csdg.py sample my_graph.json --cut cf --seeds 1,2,3 --hidden
  python csdg.py cuts hiring                            # edges each fairness cut removes

GRAPH is a built-in example key, a file saved with the page's "Download graph"
button, or a bare graph object ({"nodes": [...], "edges": [...]}).

In Python:
  import csdg
  g = csdg.load_graph('sc_marrow')
  cols, data = csdg.sample(g, rows=5000, seed=1, cut='cf')   # data[col] is a numpy array
  df = csdg.to_dataframe(cols, data)
"""
import argparse, json, math, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, 'examples')
ROLES = ['protected', 'admissible', 'outcome', 'other', 'hidden']
M32 = 0xFFFFFFFF


# ---------------- random numbers (bit-identical to the page) ----------------
def _mb32(a0, start, count):
    """Draws start..start+count-1 of mulberry32(a0), vectorized: call k sees state a0 + k*0x6D2B79F5."""
    k = np.arange(start + 1, start + count + 1, dtype=np.uint64)
    a = (np.uint64(a0 & M32) + k * np.uint64(0x6D2B79F5)) & np.uint64(M32)
    t = ((a ^ (a >> np.uint64(15))) * (a | np.uint64(1))) & np.uint64(M32)
    t = ((t + ((t ^ (t >> np.uint64(7))) * (t | np.uint64(61)))) & np.uint64(M32)) ^ t
    return ((t ^ (t >> np.uint64(14))) & np.uint64(M32)).astype(np.float64) / 4294967296.0


class Rng:
    def __init__(self, seed):
        self.a0 = int(seed) & M32
        self.k = 0

    def uniform(self, count):
        out = _mb32(self.a0, self.k, count)
        self.k += count
        return out


class Gauss:
    """makeGauss: Box-Muller pairs, cos first, sin kept as the spare for the next call."""
    def __init__(self, rng):
        self.r, self.spare = rng, None

    def draw(self, m):
        out = np.empty(m)
        i = 0
        if m and self.spare is not None:
            out[0], self.spare, i = self.spare, None, 1
        while i < m:
            pairs = (m - i + 1) // 2
            uv = self.r.uniform(2 * pairs)
            u, v = uv[0::2], uv[1::2]
            z = np.flatnonzero(u == 0)
            if len(z):  # the page redraws u until it is nonzero; replay from the first zero one draw at a time
                p = int(z[0])
                self.r.k -= 2 * (pairs - p)
                u, v = u[:p], v[:p]
            mag = np.sqrt(-2 * np.log(u))
            g = np.empty(2 * len(u))
            g[0::2], g[1::2] = mag * np.cos(2 * np.pi * v), mag * np.sin(2 * np.pi * v)
            take = min(len(g), m - i)
            out[i:i + take] = g[:take]
            if take < len(g):
                self.spare = g[take]
            i += take
            if len(z) and i < m:
                uu = 0.0
                while uu == 0:
                    uu = self.r.uniform(1)[0]
                vv = self.r.uniform(1)[0]
                mg = math.sqrt(-2 * math.log(uu))
                out[i] = mg * math.cos(2 * math.pi * vv)
                i += 1
                if i < m:
                    out[i] = mg * math.sin(2 * math.pi * vv)
                    i += 1
                else:
                    self.spare = mg * math.sin(2 * math.pi * vv)
        return out


def _int32(x):
    x = int(x) & M32
    return x - (1 << 32) if x >= 1 << 31 else x


# ---------------- distributions ----------------
def _phi(x):
    return np.exp(-x * x / 2) / math.sqrt(2 * math.pi)


def phi_inv(p):
    scalar = np.ndim(p) == 0
    p = np.atleast_1d(np.asarray(p, dtype=float))
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    pl = 0.02425
    out = np.empty_like(p)
    with np.errstate(all='ignore'):
        q = np.sqrt(-2 * np.log(p))
        lo = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        q2 = np.sqrt(-2 * np.log(1 - p))
        hi = -(((((c[0] * q2 + c[1]) * q2 + c[2]) * q2 + c[3]) * q2 + c[4]) * q2 + c[5]) / ((((d[0] * q2 + d[1]) * q2 + d[2]) * q2 + d[3]) * q2 + 1)
        q3 = p - 0.5
        r = q3 * q3
        mid = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q3 / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    out[:] = np.where(p < pl, lo, np.where(p > 1 - pl, hi, mid))
    out[p <= 0], out[p >= 1] = -8, 8
    return out[0] if scalar else out


def cat_weights(n):
    k = n['card']
    if n['shape'] == 'custom':
        p = []
        for tok in str(n.get('weights') or '').replace(';', ',').replace(',', ' ').split():
            try:
                x = float(tok)
            except ValueError:
                continue
            if math.isfinite(x) and x >= 0:
                p.append(x)
        w = [p[i] if i < len(p) else 0.0 for i in range(k)]
        if all(x == 0 for x in w):
            w = [1.0] * k
    elif n['shape'] == 'skewed':
        a = 3 * n['skew']
        w = [math.pow(i + 1, -a) for i in range(k)]
    else:
        w = [1.0] * k
    s = 0.0
    for x in w:
        s += x
    return [x / s for x in w]


def _shuffle_for(nid, k):
    r = Rng(_int32(nid * 7919 + 101))
    p = list(range(k))
    for i in range(k - 1, 0, -1):
        j = math.floor(r.uniform(1)[0] * (i + 1))
        p[i], p[j] = p[j], p[i]
    return p


def cat_bands(n):
    w, k = cat_weights(n), n['card']
    perm = list(range(k)) if n['ordinal'] else _shuffle_for(n['id'], k)
    cum = [0.0]
    for j in range(k):
        cum.append(cum[j] + w[perm[j]])
    cum[k] = 1.0
    score, v = [0.0] * k, 0.0
    for j in range(k):
        pj = cum[j + 1] - cum[j]
        m = 0.0 if pj < 1e-12 else float(_phi(phi_inv(cum[j])) - _phi(phi_inv(cum[j + 1]))) / pj
        score[perm[j]] = m
        v += pj * m * m
    sd = math.sqrt(v) or 1.0
    return np.array(cum), np.array(perm), np.array([s / sd for s in score])


_SHAPES = {
    'normal': lambda t: np.exp(-0.5 * ((t - 0.5) / 0.17) ** 2),
    'uniform': lambda t: np.ones_like(t),
    'skewed': lambda t: np.power(t, 0.7) * np.power(1 - t, 4),
    'bimodal': lambda t: np.exp(-0.5 * ((t - 0.27) / 0.09) ** 2) + np.exp(-0.5 * ((t - 0.73) / 0.09) ** 2),
}
_CDF = {}
for _k, _f in _SHAPES.items():
    _c = np.concatenate([[0.0], np.cumsum(_f((np.arange(1024) + 0.5) / 1024))])
    _CDF[_k] = _c / _c[-1]


def shape_inv(shape, u):
    c = _CDF.get(shape, _CDF['normal'])
    G = len(c) - 1
    lo = np.clip(np.searchsorted(c, u, side='right') - 1, 0, G - 1)
    span = c[lo + 1] - c[lo]
    with np.errstate(all='ignore'):
        f = np.where(span > 0, (u - c[lo]) / np.where(span > 0, span, 1), 0.5)
    return (lo + f) / G


# ---------------- graph ----------------
def _num(x, default=0.0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def normalize(g):
    """normalizeGraph + blankNode defaults: the same repairs the page makes when it loads a graph."""
    if 'graph' in g and 'nodes' not in g:
        g = g['graph']
    out = {'mode': g.get('mode') if g.get('mode') in ('mixed', 'directed', 'undirected') else 'mixed', 'nodes': [], 'edges': []}
    for n in g.get('nodes', []):
        nid = int(n['id'])
        nn = dict(name='X%d' % nid, type='cat', card=3, shape='uniform', skew=0.5, weights='', ordinal=True, pos=None,
                  cshape='normal', lo=0, hi=1, integer=False, role='other')
        nn.update(n)
        nn['id'] = nid
        nn['card'] = min(200, max(2, int(math.floor(_num(nn['card'], 2) + 0.5)) or 2))
        if nn['role'] not in ROLES:
            nn['role'] = 'other'
        if nn['type'] != 'cont':
            nn['type'] = 'cat'
        out['nodes'].append(nn)
    ids = {n['id'] for n in out['nodes']}
    for e in g.get('edges', []):
        a, b = int(e['a']), int(e['b'])
        if a not in ids or b not in ids or a == b:
            continue
        mod = e.get('mod')
        out['edges'].append(dict(id=int(e['id']), a=a, b=b, kind='und' if e.get('kind') == 'und' else 'dir',
                                 strength=min(1.0, max(0.0, _num(e.get('strength')))), sign=-1 if _num(e.get('sign'), 1) < 0 else 1,
                                 mod=int(mod) if mod is not None and int(mod) in ids else None,
                                 modOn=[int(v) for v in e['modOn']] if isinstance(e.get('modOn'), list) else None,
                                 modElse=min(1.0, max(-1.0, _num(e.get('modElse')))) if e.get('modElse') is not None else 0.0))
    return out


def load_graph(src):
    """A built-in example key, a path to a saved graph (page export or bare graph), or a dict."""
    if isinstance(src, dict):
        return normalize(src)
    path = src if os.path.exists(src) else os.path.join(EXAMPLES, src.replace(':', '_') + '.json')
    if not os.path.exists(path):
        raise SystemExit(f'No graph file or built-in example called {src!r}. Run "python csdg.py list".')
    with open(path) as f:
        return normalize(json.load(f))


def structure(g):
    """Chain components (undirected blocks) in sampling order, as structure() in the page."""
    ids = [n['id'] for n in g['nodes']]
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for e in g['edges']:
        if e['kind'] == 'und':
            parent[find(e['a'])] = find(e['b'])
    comp_map = {}
    for i in ids:
        comp_map.setdefault(find(i), []).append(i)
    comps = list(comp_map.values())
    comp_of = {i: ci for ci, c in enumerate(comps) for i in c}
    succ, indeg = [dict() for _ in comps], [0] * len(comps)
    for e in g['edges']:
        if e['kind'] == 'dir':
            ca, cb = comp_of[e['a']], comp_of[e['b']]
            if ca == cb:
                raise ValueError(f"directed edge {e['a']}->{e['b']} points into its own undirected block")
            if cb not in succ[ca]:
                succ[ca][cb] = True
                indeg[cb] += 1
    order, q = [], [i for i, d in enumerate(indeg) if d == 0]
    while q:
        c = q.pop(0)
        order.append(c)
        for d in succ[c]:
            indeg[d] -= 1
            if indeg[d] == 0:
                q.append(d)
    if len(order) < len(comps):
        raise ValueError('the directed edges form a cycle')
    return comps, order


def _cholesky(A, n):
    L = np.zeros(n * n)
    for i in range(n):
        for j in range(i + 1):
            s = A[i * n + j]
            for k in range(j):
                s -= L[i * n + k] * L[j * n + k]
            if i == j:
                if s <= 0.02:
                    return None
                L[i * n + i] = math.sqrt(s)
            else:
                L[i * n + j] = s / L[j * n + j]
    return L


def _invert(A, n):
    M = [[A[i * n + j] for j in range(n)] + [1.0 if j == i else 0.0 for j in range(n)] for i in range(n)]
    for c in range(n):
        p = c
        for r in range(c + 1, n):
            if abs(M[r][c]) > abs(M[p][c]):
                p = r
        M[c], M[p] = M[p], M[c]
        v = M[c][c]
        M[c] = [x / v for x in M[c]]
        for r in range(n):
            if r != c:
                f = M[r][c]
                if f:
                    M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return np.array([M[i][n + j] for i in range(n) for j in range(n)])


def _mod_spec(g, nodes, e):
    if e['kind'] != 'dir' or e['mod'] is None or e['mod'] == e['a']:
        return None
    M = nodes.get(e['mod'])
    if M is None or not any(x['kind'] == 'dir' and x['a'] == e['mod'] and x['b'] == e['b'] for x in g['edges']):
        return None
    L = M['card'] if M['type'] == 'cat' else 2
    on = [v for v in e['modOn'] if 0 <= v < L] if isinstance(e['modOn'], list) else [L - 1]
    act = np.zeros(L, dtype=bool)
    act[on] = True
    return M['id'], M['type'] != 'cat', act, min(1.0, max(-1.0, e['modElse'] or 0.0))


# ---------------- fairness cuts (fairCuts in the page) ----------------
def _dsep(x, y, Z, par):
    anc, st = set(), [x, y, *Z]
    while st:
        v = st.pop()
        if v in anc:
            continue
        anc.add(v)
        st.extend(par[v])
    nb = {v: set() for v in anc}
    for v in anc:
        for p in par[v]:
            nb[v].add(p)
            nb[p].add(v)
        for p in par[v]:
            for q in par[v]:
                if p != q:
                    nb[p].add(q)
    seen, q = {x}, [x]
    while q:
        v = q.pop()
        if v == y:
            return False
        for w in nb[v]:
            if w not in seen and w not in Z:
                seen.add(w)
                q.append(w)
    return True


def fair_cuts(g, rule):
    """Edge ids that a generation-time cut removes under rule 'ftu', 'cf' or 'dp'."""
    cut = set()
    if not rule:
        return cut
    dr = [e for e in g['edges'] if e['kind'] == 'dir']
    par = {n['id']: [] for n in g['nodes']}
    kid = {n['id']: [] for n in g['nodes']}
    for e in dr:
        par[e['b']].append(e['a'])
        kid[e['a']].append(e['b'])
    ids = lambda r: [n['id'] for n in g['nodes'] if n['role'] == r]
    S, Y, A = ids('protected'), ids('outcome'), set(ids('admissible'))

    def mark(a, b):
        for e in dr:
            if e['a'] == a and e['b'] == b:
                cut.add(e['id'])
                return
    if not S or not Y:
        return cut

    if rule == 'ftu':
        cut |= {e['id'] for e in dr if e['a'] in S and e['b'] in Y}
    elif rule in ('cf', 'dp'):
        # DECAF Cor. 1; DemPar is the same test with no admissible columns (DECAF Cor. 3).
        R = A if rule == 'cf' else set()
        tied = lambda v: v in S or any(not _dsep(v, s, {a for a in R if a != v}, par) for s in S)
        for y in Y:
            for u in par[y]:
                if u not in R and tied(u):
                    mark(u, y)
            for c in kid[y]:
                if c not in R and tied(c):
                    mark(y, c)
                for sp in par[c]:
                    if sp != y and sp not in R and tied(sp):
                        mark(sp, c)
    else:
        raise ValueError(f'unknown cut rule {rule!r} (use ftu, cf or dp)')
    und = [e for e in g['edges'] if e['kind'] == 'und']
    if not und:
        return cut
    if rule == 'ftu':
        cut |= {e['id'] for e in und if (e['a'] in S and e['b'] in Y) or (e['b'] in S and e['a'] in Y)}
        return cut
    block = A if rule == 'cf' else set()

    nb = {n['id']: [] for n in g['nodes']}

    def link(e):
        nb[e['a']].append(e['b'])
        nb[e['b']].append(e['a'])

    def reach(src):
        # Nodes a route from src can pass through: src, and every node reached without crossing a blocker.
        seen, st = set(src), list(src)
        while st:
            for w in nb[st.pop()]:
                if w not in seen and w not in block:
                    seen.add(w)
                    st.append(w)
        return seen
    for e in dr:
        if e['id'] not in cut:
            link(e)
    # Strongest first; A - B is dropped if one end already reaches S and the other Y through kept edges.
    for e in sorted(und, key=lambda e: (-e['strength'], e['id'])):
        fs, fy = reach(S), reach(Y)
        if (e['a'] in fs and e['b'] in fy) or (e['b'] in fs and e['a'] in fy):
            cut.add(e['id'])
        else:
            link(e)
    return cut


# ---------------- sampler (sampleData in the page) ----------------
def sample(g, rows=5000, seed=7, cut=None, hidden=False, return_warnings=False, cut_strength=1.0):
    """Returns (column names, {column: numpy array}). Categorical columns hold integer codes.

    `cut` is a rule name ('ftu', 'cf', 'dp') or a set of edge ids to cut. `cut_strength` below 1
    shuffles the cut parents for that share of the rows only (the same rows for all of a child's
    cut parents), as the experiments' partial cuts do. The page's cut strength curves use the same rows."""
    g = normalize(g)
    n = int(rows)
    nodes = {nd['id']: nd for nd in g['nodes']}
    comps, order = structure(g)
    seed0 = (int(seed) & M32) or 1
    gauss = Gauss(Rng(seed0))
    cutset = set(cut) if isinstance(cut, (set, frozenset, list, tuple)) else fair_cuts(g, cut)
    if not 0.0 <= cut_strength <= 1.0:
        raise ValueError(f'cut_strength {cut_strength} outside [0, 1]')
    codes, vals, u, score, warnings = {}, {}, {}, {}, []

    def shuffled(e, a):
        r = Rng((seed0 ^ ((e['id'] * 2654435761) & M32)) & M32)
        j = np.floor(r.uniform(n - 1) * np.arange(n, 1, -1)).astype(np.int64) if n > 1 else []
        pm = list(range(n))
        for t, i in enumerate(range(n - 1, 0, -1)):
            jj = j[t]
            pm[i], pm[jj] = pm[jj], pm[i]
        return a[np.array(pm)]

    for ci in order:
        comp = comps[ci]
        k = len(comp)
        pos = {nid: i for i, nid in enumerate(comp)}
        und = [e for e in g['edges'] if e['kind'] == 'und' and e['a'] in pos and e['b'] in pos and e['id'] not in cutset]
        L = None
        if k == 1:
            L = np.array([1.0])
        else:
            shrink = 1.0
            for _ in range(80):
                Om = np.zeros(k * k)
                for i in range(k):
                    Om[i * k + i] = 1
                for e in und:
                    i, j = pos[e['a']], pos[e['b']]
                    v = -e['sign'] * e['strength'] * 0.95 * shrink
                    Om[i * k + j] = v
                    Om[j * k + i] = v
                if _cholesky(Om, k) is not None:
                    Sg = _invert(Om, k)
                    C = np.array([Sg[i * k + j] / math.sqrt(Sg[i * k + i] * Sg[j * k + j]) for i in range(k) for j in range(k)])
                    L = _cholesky(C, k)
                    if L is None:
                        C2 = C.copy()
                        C2[::k + 1] += 0.05
                        L = _cholesky(C2, k)
                if L is not None:
                    break
                shrink *= 0.92
            if L is None:
                L = np.eye(k).ravel()
            if shrink < 0.999:
                warnings.append('block {%s}: undirected strengths scaled to %d%%' % (', '.join(nodes[i]['name'] for i in comp), round(shrink * 100)))
        Gm = gauss.draw(n * k).reshape(n, k)
        E = np.zeros((n, k))
        for i in range(k):
            s = np.zeros(n)
            for j in range(i + 1):
                s = s + L[i * k + j] * Gm[:, j]
            E[:, i] = s
        for nid in comp:
            nd, i = nodes[nid], pos[nid]
            pe = [e for e in g['edges'] if e['kind'] == 'dir' and e['b'] == nid]
            tot = 0.0
            for e in pe:
                tot += e['strength']
            sc = 0.97 / tot if tot > 0.97 else 1.0
            if tot > 0.97:
                warnings.append('%s: incoming strengths add to %.2f; scaled to 0.97' % (nd['name'], tot))
            ws = [e['sign'] * math.sqrt(e['strength'] * sc) for e in pe]
            ps = [shuffled(e, score[e['a']]) if e['id'] in cutset else score[e['a']] for e in pe]
            if cut_strength < 1.0 and any(e['id'] in cutset for e in pe):
                keep = Rng((seed0 ^ ((nid * 2246822519 + 0x5bd1e995) & M32)) & M32 or 1).uniform(n) >= cut_strength
                ps = [np.where(keep, score[e['a']], q) if e['id'] in cutset else q for e, q in zip(pe, ps)]
            md = [_mod_spec(g, nodes, e) for e in pe]
            c = math.sqrt(max(0.03, 1 - tot * sc))
            z = c * E[:, i]
            for p in range(len(pe)):
                term = ws[p] * ps[p]
                if md[p] is not None:
                    mid, cont, act, els = md[p]
                    lvl = (u[mid] >= 0.5).astype(np.int64) if cont else codes[mid]
                    term = term * np.where(act[lvl], 1.0, els)
                z = z + term
            ordr = np.argsort(z, kind='stable')
            uu = np.empty(n)
            uu[ordr] = (np.arange(n) + 0.5) / n
            u[nid] = uu
            if nd['type'] == 'cat':
                cum, perm, bscore = cat_bands(nd)
                K = nd['card']
                lo = np.clip(np.searchsorted(cum, uu, side='right') - 1, 0, K - 1)
                cd = perm[lo]
                codes[nid] = cd
                score[nid] = bscore[cd]
            else:
                lo, hi = float(nd['lo']), float(nd['hi'])
                x = lo + (hi - lo) * shape_inv(nd['cshape'], uu)
                if nd['integer']:
                    x = np.floor(x + 0.5)
                vals[nid] = x
                score[nid] = phi_inv(uu)
    cols, data, used = [], {}, set()
    for nd in g['nodes']:
        if nd['role'] == 'hidden' and not hidden:
            continue
        nm, i = nd['name'], 2
        while nm in used:
            nm, i = '%s_%d' % (nd['name'], i), i + 1
        used.add(nm)
        cols.append(nm)
        data[nm] = codes[nd['id']] if nd['type'] == 'cat' else vals[nd['id']]
    return (cols, data, warnings) if return_warnings else (cols, data)


# ---------------- output ----------------
def _js_num(v):
    """Number(v.toPrecision(6)) printed the way JavaScript prints numbers."""
    v = float('%.6g' % v)
    if v == int(v) and abs(v) < 1e21:
        return str(int(v))
    s = repr(v)
    if 'e' in s:
        m, ex = s.split('e')
        ex = int(ex)
        if -7 < ex < 21:
            from decimal import Decimal
            s = format(Decimal(s), 'f')
        else:
            s = '%se%s%d' % (m, '+' if ex > 0 else '-', abs(ex))
    return s


def to_csv(cols, data, g=None, path=None):
    """The page's CSV text: integer codes for categories, 6 significant digits otherwise."""
    fmt = []
    for c in cols:
        a = data[c]
        if a.dtype.kind in 'iu':
            fmt.append([str(int(x)) for x in a])
        else:
            fmt.append([_js_num(x) for x in a])
    lines = [','.join(cols)] + [','.join(r) for r in zip(*fmt)]
    text = '\n'.join(lines) + '\n'
    if path:
        with open(path, 'w') as f:
            f.write(text)
    return text


def to_dataframe(cols, data):
    import pandas as pd
    return pd.DataFrame({c: data[c] for c in cols}, columns=cols)


def describe(g, cut=None):
    """Roles and the cut edges by name, for a manifest next to the CSVs."""
    name = {nd['id']: nd['name'] for nd in g['nodes']}
    by = lambda r: [nd['name'] for nd in g['nodes'] if nd['role'] == r]
    return {'roles': {r: by(r) for r in ROLES},
            'labels': {nd['name']: nd['labels'] for nd in g['nodes'] if nd.get('labels')},
            'cuts': {r: sorted([name[e['a']], name[e['b']]] for e in g['edges'] if e['id'] in fair_cuts(g, r)) for r in ('ftu', 'cf', 'dp')},
            'applied_cut': cut or None}


def _seeds(spec):
    out = []
    for part in str(spec).split(','):
        if '-' in part.strip()[1:]:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description='Sample tables from Causal Synthetic Data Generator graphs without a browser.')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('list', help='list built-in examples')
    s = sub.add_parser('sample', help='write one CSV per graph and seed')
    s.add_argument('graphs', nargs='+', help='example keys or graph JSON files')
    s.add_argument('--rows', type=int, default=5000)
    s.add_argument('--seeds', default='7', help='e.g. 7 or 1-20 or 1,5,9 (default 7, the page default)')
    s.add_argument('--cut', choices=['none', 'ftu', 'cf', 'dp'], default='none', help='fairness cut applied at generation')
    s.add_argument('--hidden', action='store_true', help='also write hidden columns (the page leaves them out)')
    s.add_argument('--out', default='.', help='output folder')
    c = sub.add_parser('cuts', help='print the edges each fairness cut removes')
    c.add_argument('graph')
    a = ap.parse_args(argv)
    if a.cmd == 'list':
        for f in sorted(os.listdir(EXAMPLES)):
            if f.endswith('.json'):
                g = load_graph(os.path.join(EXAMPLES, f))
                k = re.sub(r'^(dsg?)_', r'\1:', f[:-5])
                obs = sum(nd['role'] != 'hidden' for nd in g['nodes'])
                print('%-18s %3d nodes (%d in the table), %3d edges' % (k, len(g['nodes']), obs, len(g['edges'])))
        return
    if a.cmd == 'cuts':
        g = load_graph(a.graph)
        for r, pairs in describe(g)['cuts'].items():
            print('%-3s %s' % (r, ', '.join('%s->%s' % tuple(p) for p in pairs) or '(none)'))
        return
    os.makedirs(a.out, exist_ok=True)
    cut = None if a.cut == 'none' else a.cut
    for src in a.graphs:
        g = load_graph(src)
        stem = os.path.splitext(os.path.basename(src))[0].replace(':', '_') + ('' if cut is None else '_' + cut)
        seeds = _seeds(a.seeds)
        for sd in seeds:
            cols, data, warn = sample(g, a.rows, sd, cut, a.hidden, return_warnings=True)
            p = os.path.join(a.out, '%s_seed%d.csv' % (stem, sd))
            to_csv(cols, data, g, p)
        man = dict(describe(g, cut), source=src, rows=a.rows, seeds=seeds, hidden_included=a.hidden, columns=cols, warnings=warn, graph=g)
        with open(os.path.join(a.out, stem + '_manifest.json'), 'w') as f:
            json.dump(man, f, indent=1)
        print('%s: %d CSVs of %d rows x %d columns in %s' % (src, len(seeds), a.rows, len(cols), a.out) + (('; ' + '; '.join(warn)) if warn else ''))


if __name__ == '__main__':
    main()
