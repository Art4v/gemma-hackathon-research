"""Design statistics of training targets: how matmul kernels are tiled, by operand size.

For every row, the final assistant message is the target. The reference in the first
user message is exec'd-free: shapes are read from the CONFIG literal only as a proxy
(largest integer product is not reliable), so sizes come from the verdict files when
given, otherwise from regexes on the target.
"""
import collections
import json
import re
import statistics
import sys

path = sys.argv[1]
rows = [json.loads(l) for l in open(path)]


def target(r):
    return r["messages"][-1]["content"]


def is_matmul(code):
    return bool(re.search(r"jnp\.dot\(|jnp\.matmul\(|dot_general\(|jnp\.einsum\(|lax\.dot\(| @ ", code))


def grid_rank(code):
    m = re.search(r"grid\s*=\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)", code)
    if not m:
        return None
    inner = m.group(1).strip().rstrip(",")
    depth, n = 0, 1
    for ch in inner:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            n += 1
    return n if inner else 0


kinds = collections.Counter(r.get("kind") for r in rows)
print(path.split("/")[-1], len(rows), dict(kinds))
mm = [r for r in rows if is_matmul(target(r))]
print(f"targets with a matmul in them: {len(mm)} ({100*len(mm)/len(rows):.0f}%)")
acc = [r for r in mm if "scratch_shapes" in target(r) and "program_id" in target(r)]
print(f"  of those, K-accumulating (scratch + program_id): {len(acc)} ({100*len(acc)/max(1,len(mm)):.0f}%)")
ranks = collections.Counter(grid_rank(target(r)) for r in mm)
print("  grid rank among matmul targets:", dict(sorted(ranks.items(), key=lambda kv: str(kv[0]))))
vm = [r for r in rows if "vmem_limit_bytes" in target(r)]
print(f"targets setting vmem_limit_bytes: {len(vm)} ({100*len(vm)/len(rows):.0f}%)")
vals = collections.Counter()
for r in vm:
    for m in re.finditer(r"vmem_limit_bytes\s*=\s*([^,\n)]+(?:\)[^,\n)]*)?)", target(r)):
        vals[m.group(1).strip()[:40]] += 1
print("  most common values:", vals.most_common(8))
fallback = [r for r in rows if re.search(r"if\s+\w+\s*%\s*\d+\s*==\s*0\s+else\s+\w+\b", target(r))]
print(f"targets with an 'X if n % X == 0 else <whole dim>' block choice: {len(fallback)} ({100*len(fallback)/len(rows):.0f}%)")
lit = collections.Counter()
for r in mm:
    for m in re.finditer(r"\b(\d{2,5})\s+if\s+\w+\s*%\s*\1", target(r)):
        lit[int(m.group(1))] += 1
print("  literal tile sizes in matmul targets (X if n % X):", sorted(lit.items()))
lens = [len(target(r)) for r in rows]
print("target chars: median", int(statistics.median(lens)), "p90", sorted(lens)[int(0.9 * len(lens))])
conv = [r for r in rows if re.search(r"conv", r.get("task", "") + r.get("problem_id", ""), re.I)]
att = [r for r in rows if re.search(r"attn|attention", r.get("task", "") + r.get("problem_id", ""), re.I)]
print(f"task-name families: conv {len(conv)}, attention {len(att)}")
src = collections.Counter((r.get("source") or (r.get("problem_id") or r.get("task") or "?").split("/")[0].split("__")[0]) for r in rows)
print("sources:", src.most_common(8))
