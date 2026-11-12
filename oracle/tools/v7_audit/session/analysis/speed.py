"""Speed and design of v6 targets, split by training set and by whether the kernel has a matmul."""
import json
import re
import statistics
import sys

rows = [json.loads(l) for l in open(sys.argv[1])]


def tgt(r):
    return r["messages"][-1]["content"]


def mm(code):
    return bool(re.search(r"jnp\.dot\(|jnp\.matmul\(|dot_general\(|jnp\.einsum\(|lax\.dot\(| @ ", code))


def summarise(name, rs):
    sp = [r["speed_vs_xla"] for r in rs if isinstance(r.get("speed_vs_xla"), (int, float))]
    if not sp:
        print(f"{name}: n={len(rs)} no speeds")
        return
    sp.sort()
    q = lambda p: sp[min(len(sp) - 1, int(p * len(sp)))]
    slow = sum(s < 0.7 for s in sp)
    print(f"{name}: n={len(rs)} speed p10={q(.1):.2f} median={q(.5):.2f} p90={q(.9):.2f} | below 0.7x: {slow} ({100*slow/len(sp):.0f}%) | below 0.3x: {sum(s<0.3 for s in sp)}")


summarise("all", rows)
summarise("matmul targets", [r for r in rows if mm(tgt(r))])
summarise("matmul, K-accumulating", [r for r in rows if mm(tgt(r)) and "scratch_shapes" in tgt(r) and "program_id" in tgt(r)])
summarise("non-matmul targets", [r for r in rows if not mm(tgt(r))])
for s in sorted({r.get("set") for r in rows}):
    summarise(f"set={s}", [r for r in rows if r.get("set") == s])
print("imports pltpu:", sum("pltpu" in tgt(r) for r in rows), "of", len(rows))
print("uses CompilerParams:", sum("CompilerParams" in tgt(r) for r in rows), "| dimension_semantics:", sum("dimension_semantics" in tgt(r) for r in rows))
ex = [r for r in rows if r.get("set") == "e3_scale" and mm(tgt(r)) and r["kind"] == "single"]
print("\n=== one e3_scale matmul target:", ex[0]["problem_id"], "speed", ex[0]["speed_vs_xla"], "scaled_from", ex[0].get("scaled_from"))
u = ex[0]["messages"][1]["content"]
m = re.search(r"CONFIG = \{.*?\n\}", u, re.S)
print(m.group(0) if m else u[:600])
print(tgt(ex[0])[:2600])
