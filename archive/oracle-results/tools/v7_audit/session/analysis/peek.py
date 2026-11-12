"""Summarise the v4 JAXBench run: row schema, per-task verdict by turn, error heads."""
import collections
import json
import sys

run = sys.argv[1]
rows = [json.loads(l) for l in open(f"{run}/bare.jsonl")]
verd = [json.loads(l) for l in open(f"{run}/verdicts.jsonl")]
print("row keys:", sorted(rows[0].keys()))
print("verdict keys:", sorted(verd[0].keys()))
print("n rows", len(rows), "n verdicts", len(verd))
r0 = rows[0]
for k, v in r0.items():
    s = json.dumps(v) if not isinstance(v, str) else v
    print(f"  {k}: {s[:160]!r}")
v0 = verd[0]
for k, v in v0.items():
    s = json.dumps(v) if not isinstance(v, str) else v
    print(f"  V {k}: {s[:200]!r}")
