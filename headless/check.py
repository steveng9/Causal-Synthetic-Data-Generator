#!/usr/bin/env python3
"""Check that csdg.py writes byte-identical CSVs to the page (hashes from export_from_page.py)."""
import hashlib, json, os, sys
import csdg

HERE = os.path.dirname(os.path.abspath(__file__))
refs = json.load(open(os.path.join(HERE, 'reference_hashes.json')))
bad = []
for r in refs:
    g = csdg.load_graph(r['example'])
    cols, data = csdg.sample(g, r['rows'], r['seed'], r['cut'])
    if hashlib.sha256(csdg.to_csv(cols, data, g).encode()).hexdigest() != r['sha256']:
        bad.append('%(example)s rows=%(rows)d seed=%(seed)d cut=%(cut)s' % r)
print('%d of %d tables match the page exactly' % (len(refs) - len(bad), len(refs)))
for b in bad:
    print('  differs:', b)
sys.exit(1 if bad else 0)
