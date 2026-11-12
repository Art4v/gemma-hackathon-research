"""For v4's turn-0 replies on matmul-led k-tasks: which tiling design did it choose, and what happened."""
import collections
import json
import re
import sys

run = sys.argv[1]
rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{run}/bare.jsonl")}
verd = {}
for l in open(f"{run}/verdicts.jsonl"):
    v = json.loads(l)
    verd[v["id"]] = v


def grid_rank(code):
    m = re.search(r"grid\s*=\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)", code)
    if not m:
        return None
    inner, depth, n = m.group(1).strip().rstrip(","), 0, 1
    for ch in inner:
        depth += ch in "(["
        depth -= ch in ")]"
        n += ch == "," and depth == 0
    return n


tab = collections.Counter()
for i, r in sorted(rows.items()):
    t = r["task"]
    if not i.endswith("/t0") or not re.match(r"\d+k_(Matmul|Gemm|BMM)", t):
        continue
    code = r["code"] or ""
    acc = "scratch_shapes" in code and "program_id" in code
    lim = re.search(r"vmem_limit_bytes\s*=\s*([^,\n)]+)", code)
    v = verd[i]["verdict"]
    design = "K-accumulating" if acc else "whole-K block"
    tab[(design, v)] += 1
    print(f"{t[:42]:42s} grid={grid_rank(code)} {design:15s} limit={(lim.group(1).strip() if lim else 'default'):18s} -> {v}")
print()
for (d, v), n in sorted(tab.items()):
    print(f"{d:15s} {v:13s} {n}")
