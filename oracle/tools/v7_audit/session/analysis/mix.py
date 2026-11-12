"""Matmul design mix per row kind: whole-K block vs K-accumulating, for base-set options."""
import collections
import json
import re
import sys


def tgt(r):
    return r["messages"][-1]["content"]


def mm(code):
    return bool(re.search(r"jnp\.dot\(|jnp\.matmul\(|dot_general\(|jnp\.einsum\(|lax\.dot\(| @ ", code))


def kacc(code):
    return "scratch_shapes" in code and "program_id" in code


for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path)]
    tab = collections.defaultdict(lambda: [0, 0, 0])
    for r in rows:
        c = tgt(r)
        k = r.get("kind")
        tab[k][0] += 1
        if mm(c):
            tab[k][2 if kacc(c) else 1] += 1
    print(path.split("/")[-1])
    print(f"  {'kind':16s} rows  matmul:whole-K  matmul:K-acc")
    tot = [0, 0, 0]
    for k, (n, w, a) in sorted(tab.items()):
        print(f"  {k:16s} {n:5d} {w:8d} {a:12d}")
        tot = [tot[0] + n, tot[1] + w, tot[2] + a]
    print(f"  {'TOTAL':16s} {tot[0]:5d} {tot[1]:8d} {tot[2]:12d}  whole-K share of matmul = {100*tot[1]/(tot[1]+tot[2]):.0f}%")
