"""Show the schema and one example of a training jsonl (messages truncated)."""
import collections
import json
import sys

path = sys.argv[1]
rows = [json.loads(l) for l in open(path)]
print(path.split("/")[-1], "rows:", len(rows))
print("keys:", sorted(rows[0].keys()))
r = rows[0]
for k, v in r.items():
    if k == "messages":
        for m in v:
            c = m["content"]
            print(f"  [{m['role']}] ({len(c)} chars) {c[:1500]!r}")
    else:
        print(f"  {k}: {json.dumps(v)[:300]}")
meta_keys = collections.Counter()
for r in rows:
    for k in (r.get("meta") or {}):
        meta_keys[k] += 1
print("meta keys:", dict(meta_keys))
