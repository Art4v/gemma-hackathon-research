"""v4-solved matmul problems that the v7 list (built from v6 strict only) never considered."""
import collections
import json
import re
import sys

K, V4 = sys.argv[1], sys.argv[2]
J = lambda p: [json.loads(l) for l in open(p)]
L = lambda f: {l.strip() for l in open(f"{K}/kimi/v6/{f}") if l.strip() and not l.startswith("#")}
mm = lambda code: bool(re.search(r"jnp\.dot\(|jnp\.matmul\(|dot_general\(|jnp\.einsum\(|lax\.dot\(| @ |jnp\.tensordot\(", code))
ref_of = lambda r: next(m["content"] for m in r["messages"] if m["role"] == "user")

v4 = J(V4)
print("v4 sample:", {k: (str(v)[:70]) for k, v in v4[0].items() if k not in ("messages", "thinking")})
print("v4 teachers", collections.Counter(r.get("teacher") for r in v4).most_common(8))
strict = J(f"{K}/kimi/v6/v6_train_strict.jsonl")
print("v6 sample:", {k: (str(v)[:70]) for k, v in strict[0].items() if k not in ("messages", "thinking")})
pool = {}
for f in ("data/full_problems.jsonl", "data/new/new_problems.jsonl"):
    for r in J(f"{K}/kimi/{f}"):
        pool[r["problem_id"]] = r
print("pool", len(pool), "sample keys", sorted(next(iter(pool.values())).keys()))
byname = collections.defaultdict(set)
for p, r in pool.items():
    byname[r.get("task_name")].add(p)
sp = {r["problem_id"] for r in strict}
flag = set().union(*(L(f) for f in ("exclude_hard.txt", "flag_unsure.txt", "rule3_families.txt", "flag_generic_stacks.txt", "flag_generic_stacks_new.txt")))

v4t = collections.defaultdict(list)
for r in v4:
    v4t[r["task"]].append(r)
single = {t for t, rs in v4t.items() if any(x["kind"] in ("single", "repair", "generated", "shape_variant") for x in rs)}
v4mm = {t for t in single if mm(ref_of(v4t[t][0]))}
print("v4 tasks", len(v4t), "| with a verified kernel row", len(single), "| matmul in ref", len(v4mm))
inpool = {t for t in v4mm if t in byname or t in pool}
print("  v4 matmul tasks that map to a pool problem:", len(inpool))
ids = lambda t: byname.get(t, set()) | ({t} if t in pool else set())
not_v6 = {t for t in v4mm if not (ids(t) & sp)}
print("  v4 matmul tasks with NO v6-strict row:", len(not_v6), "| of those flagged:", len({t for t in not_v6 if ids(t) & flag}), "| unmapped:", len({t for t in not_v6 if not ids(t)}))
print("  sample:", sorted(not_v6)[:8])
print("  sources:", collections.Counter(v4t[t][0]["source"] for t in not_v6).most_common())
