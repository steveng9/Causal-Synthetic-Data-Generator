#!/usr/bin/env python3
"""Run index.html's own code without a browser (gjs or node) to refresh the files csdg.py relies on.

  python export_from_page.py            # rewrites examples/*.json and reference_hashes.json

Run it after changing the page's sampler or its built-in examples, then run check.py.
The browser parts of the page are replaced by a do-nothing stand-in; the sampler,
templates and CSV writer run unchanged.
"""
import glob, hashlib, json, os, shutil, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, '..', 'index.html')
CASES = [(400, 3, ''), (300, 11, 'cf'), (300, 5, 'dp'), (300, 8, 'ftu')]  # rows, seed, cut

STUB = r'''
const _h = { get(t, k) { if (k === Symbol.iterator) return function* () {}; if (k === Symbol.toPrimitive) return () => 0;
  if (k === 'length') return 0; if (k === 'value') return ''; if (k === 'then') return () => S; return S; }, apply() { return S; }, set() { return true; } };
const S = new Proxy(function () {}, _h);
var document = S, window = S, navigator = S, localStorage = {getItem: () => null, setItem() {}}, location = {search: ''};
var setTimeout = () => 0, clearTimeout = () => 0, requestAnimationFrame = () => 0, getComputedStyle = () => S, matchMedia = () => S;
var ResizeObserver = function () { return S; }, MutationObserver = ResizeObserver, IntersectionObserver = ResizeObserver;
var URLSearchParams = function () { return {get: () => null}; };
var performance = {now: () => Date.now()};
'''
DRIVER = r'''
const IS_NODE = typeof process !== 'undefined' && !!(process.versions && process.versions.node);
const OUT = IS_NODE ? process.argv[2] : ARGV[0];
const write = IS_NODE ? (p, t) => require('fs').writeFileSync(OUT + '/' + p, t) : (p, t) => imports.gi.GLib.file_set_contents(OUT + '/' + p, t);
const CASES = %s;
const keys = Object.keys(TEMPLATES).filter(k => k !== 'empty').concat(Object.keys(DATASETS).map(k => 'ds:' + k));
for (const k of keys) {
  if (k.startsWith('ds:')) loadDataset(k.slice(3)); else TEMPLATES[k]();
  const f = k.replace(':', '_');
  write('ex_' + f + '.json', JSON.stringify(G));
  for (const [n, sd, cv] of CASES) { cutView = cv; DATA = sampleData(n, sd); if (!DATA.error) write(`ref_${f}_${n}_${sd}_${cv || 'none'}.csv`, toCSV()); }
  cutView = '';
}
'''


def main():
    html = open(PAGE).read()
    s = html[html.index('<script>\n') + 9: html.rindex('</script>')]
    i = s.rindex('})();')  # the page script is one closure; the driver runs inside it
    js = STUB + s[:i] + '\n;' + DRIVER % json.dumps(CASES) + '\n' + s[i:]
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, 'run.js')
        open(path, 'w').write(js)
        eng = shutil.which('gjs') or shutil.which('node')
        if not eng:
            sys.exit('Needs gjs or node on the PATH.')
        subprocess.run([eng, path, tmp], check=True)
        ex = os.path.join(HERE, 'examples')
        os.makedirs(ex, exist_ok=True)
        for f in glob.glob(os.path.join(ex, '*.json')):
            os.remove(f)
        for f in sorted(glob.glob(os.path.join(tmp, 'ex_*.json'))):
            g = json.load(open(f))
            json.dump(g, open(os.path.join(ex, os.path.basename(f)[3:]), 'w'), indent=1)
        refs = []
        for f in sorted(glob.glob(os.path.join(tmp, 'ref_*.csv'))):
            key, n, sd, cut = os.path.basename(f)[4:-4].rsplit('_', 3)
            refs.append(dict(example=key.replace('ds_', 'ds:', 1), rows=int(n), seed=int(sd), cut=None if cut == 'none' else cut,
                             sha256=hashlib.sha256(open(f, 'rb').read()).hexdigest()))
        json.dump(refs, open(os.path.join(HERE, 'reference_hashes.json'), 'w'), indent=1)
        print('%d examples, %d reference tables (%s)' % (len(glob.glob(os.path.join(ex, '*.json'))), len(refs), os.path.basename(eng)))
    finally:
        shutil.rmtree(tmp)


if __name__ == '__main__':
    main()
