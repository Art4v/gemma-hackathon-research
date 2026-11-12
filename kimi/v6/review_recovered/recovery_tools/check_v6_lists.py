"""Break down the 'keep-excluded/unsure in exclude_hard.txt U flag_unsure.txt' check, and see whether any
keep-excluded/unsure problem reaches the current v6 training files or the current Kimi pool file.
Run: python3 -I check_v6_lists.py
"""
import collections, json, os

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K = "/Users/hrudaynara/Projects/gemma-work/wt-kimi/kimi"
V = [json.loads(l) for l in open(OUT + "/regen/review/adv/adversarial_verdicts.jsonl")]
ids = lambda f: {l.strip() for l in open(f"{K}/v6/{f}") if l.strip() and not l.lstrip().startswith("#")}
hard, uns = ids("exclude_hard.txt"), ids("flag_unsure.txt")
lines = []
c = collections.Counter()
for v in V:
    if v["verdict"] == "safe to re-admit": continue
    where = "exclude_hard" if v["problem_id"] in hard else ("flag_unsure" if v["problem_id"] in uns else "neither")
    c[(v["verdict"], "in_pool" if v["in_kimi_pool_now"] else "not_in_pool", where)] += 1
for k in sorted(c): lines.append(f"{k}: {c[k]}")
keep = {v["problem_id"] for v in V if v["verdict"] != "safe to re-admit"}
def base(pid):  # scaled/mutated children carry the parent id as a prefix (e.g. __scale)
    return pid.split("__scale")[0]
for f in ("v6/v6_train.jsonl", "v6/v6_train_strict.jsonl", "data/full_problems.jsonl"):
    pids = collections.Counter()
    for l in open(f"{K}/{f}"):
        r = json.loads(l); pids[r.get("problem_id")] += 1
    hit = sorted(p for p in pids if p in keep or base(p) in keep)
    lines.append(f"{f}: {len(pids)} distinct problem_ids; keep-excluded/unsure present: {len(hit)} {hit}")
print("\n".join(lines))
with open(OUT + "/checks/check_v6_lists.log", "w") as fh:
    fh.write("\n".join(lines) + "\n")
