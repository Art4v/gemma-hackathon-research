"""Where the v7 scaled list loses candidates, and how many more matmul problems exist outside v6 strict."""
import collections
import json
import re
import sys

K = sys.argv[1]   # kimi-k31 worktree
V4 = sys.argv[2]  # v4 train.jsonl
J = lambda p: [json.loads(l) for l in open(p)]
L = lambda f: {l.strip() for l in open(f"{K}/kimi/v6/{f}") if l.strip() and not l.startswith("#")}

log = J(f"{K}/kimi/v7/kn_log.jsonl")
print("kn_log rows", len(log), "unique problems", len({r["problem_id"] for r in log}))
by = {}
for r in log:
    by[r["problem_id"]] = r
c = collections.Counter()
for r in by.values():
    if r.get("ok"):
        if r.get("factor_features") == 1 and r.get("batch_divisor") == 1:
            c["ok: already exam scale"] += 1
        elif r.get("dot_share", 0) < 0.5:
            c["ok but matmul share < 0.5"] += 1
        else:
            c["ok: scaled"] += 1
    else:
        c["fail: " + re.sub(r"\d+", "#", r.get("reason", "?"))[:60]] += 1
for k, v in c.most_common():
    print(f"  {v:4d}  {k}")
share = [r["dot_share"] for r in by.values() if not r.get("ok") and "dot_share" in r]
print("  failed-but-has-matmul:", len(share), "of which dot_share >= 0.5:", sum(s >= 0.5 for s in share))

flags = {f: L(f) for f in ("exclude_hard.txt", "flag_unsure.txt", "rule3_families.txt", "flag_generic_stacks.txt", "flag_generic_stacks_new.txt")}
a6 = L("a6_dependent_problem_ids.txt")
print("flag list sizes", {k: len(v) for k, v in flags.items()}, "a6_dependent", len(a6))

mm = lambda code: bool(re.search(r"jnp\.dot\(|jnp\.matmul\(|dot_general\(|jnp\.einsum\(|lax\.dot\(| @ |jnp\.tensordot\(", code))
ref_of = lambda r: next(m["content"] for m in r["messages"] if m["role"] == "user")
strict = J(f"{K}/kimi/v6/v6_train_strict.jsonl")
full = J(f"{K}/kimi/v6/v6_train.jsonl")
sp = {r["problem_id"] for r in strict}
print("v6 strict rows", len(strict), "problems", len(sp), "| v6 full rows", len(full), "problems", len({r['problem_id'] for r in full}))
for name, fl in flags.items():
    hit = {r["problem_id"] for r in strict if r["problem_id"] in fl}
    print(f"  v6-strict problems on {name}: {len(hit)} (matmul in ref: {len({r['problem_id'] for r in strict if r['problem_id'] in fl and mm(ref_of(r))})})")
a6p = {r["problem_id"] for r in full if r["problem_id"] not in sp}
print("  v6-full-only (A6-dependent) problems:", len(a6p), "matmul in ref:", len({r["problem_id"] for r in full if r["problem_id"] in a6p and mm(ref_of(r))}))

v4 = J(V4)
keys = sorted(v4[0].keys())
print("v4 row keys", keys)
pidk = "problem_id" if "problem_id" in v4[0] else ("task_name" if "task_name" in v4[0] else None)
v4p = collections.defaultdict(list)
for r in v4:
    v4p[r.get(pidk)].append(r)
v4mm = {p for p, rs in v4p.items() if any(mm(ref_of(r)) for r in rs)}
print("v4 problems", len(v4p), "with matmul in ref", len(v4mm), "| of those not in v6 strict:", len(v4mm - sp))
kinds = collections.Counter(r["kind"] for p in (v4mm - sp) for r in v4p[p])
print("  kinds of v4-only matmul rows", dict(kinds))
print("  sample v4 ids", list(v4mm - sp)[:5], "| sample v6 ids", list(sp)[:3])
