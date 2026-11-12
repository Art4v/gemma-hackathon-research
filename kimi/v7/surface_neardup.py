"""The team's layer-3 contamination filter (aarav/dataset-v4 dataset/scripts/filter_neardup.py: near-duplicate text against the 73 exam
baseline files) on the NEW reference text of surface_gen.py's variants, with the original reference scored alongside for comparison.
  python3 surface_neardup.py <aarav/dataset-v4 dataset/scripts dir> [gen|layout|rowout]   ->  <gen_surface|layout_surface|rowout_surface>/neardup.json"""
import json, sys, collections, statistics, pathlib
sys.path.insert(0, sys.argv[1])
import filter_neardup as fn  # noqa: E402
V = pathlib.Path(__file__).resolve().parent
D = V / {"gen": "gen_surface", "layout": "layout_surface", "rowout": "rowout_surface"}[sys.argv[2] if len(sys.argv) > 2 else "gen"]
corpus = fn.exam_corpus()
print('exam files', len(corpus))
def ref(r):
    u = r['messages'][1]['content']; return u[u.index('```python\n') + 10:u.rindex('```')]
def score(code):
    return fn.nearest(fn.normalise(code), corpus, k=1)[0]
var = [json.loads(l) for l in open(D / 'variants.jsonl')]
bad, rows = [], []
for v in var:
    so = score(v['_proof']['orig_ref']); sv = score(ref(v))
    rows.append((v['example_id'], v['family'], so['jaccard'], so['ratio'], sv['jaccard'], sv['ratio'], sv['task'], v['surface']['names'], v['is_dev']))
    if sv['jaccard'] >= fn.JACCARD_BAR or sv['ratio'] >= fn.RATIO_BAR:
        bad.append(rows[-1])
print('originals: max J %.3f max ratio %.3f | median ratio %.3f' % (max(r[2] for r in rows), max(r[3] for r in rows), statistics.median(r[3] for r in rows)))
print('variants : max J %.3f max ratio %.3f | median ratio %.3f' % (max(r[4] for r in rows), max(r[5] for r in rows), statistics.median(r[5] for r in rows)))
print('variants over the bar (J>=0.5 or ratio>=0.8):', len(bad), 'of', len(rows), '| of train rows:', sum(not b[-1] for b in bad))
print(' by names:', collections.Counter(str(b[7]) for b in bad).most_common(8))
print(' nearest exam tasks:', collections.Counter(b[6] for b in bad).most_common(8))
json.dump([{'example_id': r[0], 'orig_j': r[2], 'orig_ratio': r[3], 'j': r[4], 'ratio': r[5], 'nearest': r[6], 'excluded': (r[4] >= fn.JACCARD_BAR or r[5] >= fn.RATIO_BAR)}
           for r in rows], open(D / 'neardup.json', 'w'), indent=0)
