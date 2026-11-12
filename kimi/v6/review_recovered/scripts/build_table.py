import json, re, os, collections, sys
S = "/private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad"
sys.path.insert(0, S + "/review")
from labels import L
KIMI = set(json.load(open(f"{S}/review/replica_out_ids.json"))["out_ids"])
rows = collections.OrderedDict()
def add(pid, source, rule):
    r = rows.setdefault(pid, {"problem_id": pid, "source": source, "rules": []})
    if rule not in r["rules"]:
        r["rules"].append(rule)
# 1. KernelBench L2 exam stems (never ported)
BENCH = "/Users/hrudaynara/Projects/gemma-hackathon-research/benchmarks/jaxbench/level1"
ktask = {}
for d in sorted(os.listdir(BENCH)):
    if re.match(r"\d+k_", d):
        m = re.search(r"'name':\s*'([^']+)'", open(f"{BENCH}/{d}/baseline.py").read())
        ktask["kb/l2/" + m.group(1)] = d
for pid, task in ktask.items():
    add(pid, "kernelbench_l2 (never ported)", "L1 exam KernelBench stem: port_kernelbench.pick() skip + filter_contamination.name_match")
    L.setdefault(pid, ("A", "", task, "k_task_itself", "this IS the KernelBench L2 problem JAXBench ported as " + task))
# 2. P_TASK_TWINS
for pid in [p for p in L if p.startswith("kb/l1/") and not p.startswith("kb/l1/50_")]:
    add(pid, "kernelbench_l1 (never ported)", "L0 P_TASK_TWINS hand list (dataset/scripts/port_kernelbench.py:25-30)")
# 3. algorithm reviews
F = f"{S}/kbk-r1-wt/dataset/kernelbook"
for b in ["kbk", "kbkc", "kbkd"]:
    for l in open(f"{F}/algo_review_{b}.jsonl"):
        r = json.loads(l)
        if r["decision"] == "exclude":
            src = "prod (github/authored-from-paper)" if r["batch"] == "prod" else f"kernelbook ({r['batch']})"
            add(r["problem_id"], src, f"rule3 algorithm review algo_review_{b}.jsonl verdict={r['verdict']}")
# 4. score filters in the canonical pools
for f, tag in [(f"{S}/review/karthik_kbk_problems.jsonl", "karthik/kbk@036fe89"), (f"{S}/review/aarav_kbkc_problems.jsonl", "aarav/kbk-c")]:
    for l in open(f):
        r = json.loads(l); e = r.get("exclude_reason") or ""
        if r["status"] == "excluded" and r.get("batch") in ("kbk", "kbkd", "kbkc", "prod") and e.startswith("contamination") and not e.startswith("contamination:algorithm"):
            layer = e.split(":")[1]
            name = {"extra": "filter_extra op-multiset/val", "opchain": "L2 op-chain equality", "neardup": "L3 text near-dup"}[layer]
            add(r["problem_id"], f"kernelbook ({r['batch']})", f"{name} [{tag}] {e[len('contamination:'):][:70]}")
# 5. dataset-v1 (gemma-dataset-claude) contamination rows
for l in open("/Users/hrudaynara/Projects/gemma-dataset-claude/dataset/problems/problems.jsonl"):
    r = json.loads(l); e = r.get("exclude_reason") or ""
    if r["status"] == "excluded" and e.startswith("contamination"):
        name = "L3 text near-dup" if ":neardup:" in e else "op-multiset review tier (>=0.8), dropped by decision #5"
        add(r["problem_id"], "kernelbench_l2 port (dataset-v1)", f"{name} [gemma-dataset-claude] {e[len('contamination:'):][:70]}")
# 6. Gemini-era pool near-dups
for l in open("/Users/hrudaynara/Projects/gemma-dataset-gen/dataset/problems/problems.jsonl"):
    r = json.loads(l); e = r.get("exclude_reason") or ""
    if r["status"] == "excluded" and e.startswith("contamination"):
        add(r["problem_id"], "gemini-era pool (gemma-dataset-gen)" if r["problem_id"] not in rows else rows[r["problem_id"]]["source"], f"L3 text near-dup [gemma-dataset-gen] {e[len('contamination:'):][:70]}")
# 7. prod candidates dropped before authoring
for l in open(f"{F}/prod_excluded_by_algorithm.jsonl"):
    r = json.loads(l)
    add("prod/" + r["stem"], "prod candidate (never authored)", "pre-authoring algorithm cut prod_excluded_by_algorithm.jsonl: " + r["reason"])
# 8. v4 exclude_families
fams = [x.strip() for x in open(f"{S}/review/v4_exclude_families.txt") if x.strip()]
for pid in [p for p in L if any(p == f or p.startswith(f + "/") or p.startswith(f + "_") for f in fams)]:
    add(pid, "authored/scraped (dataset-v1) or m1 child", "v4 rule-3 family drop dataset/v4/exclude_families.txt (aarav/dataset-v4 55e432a, 6e18948)")
# 9. val exact (build_full)
for l in open(f"{S}/diag/prompts_val.jsonl"):
    r = json.loads(l); pid = r["problem_id"]
    add(pid, "val (dataset-v1)", "val exact task_name drop in kimi_pilot/build_full.py:23 (23 problems x 5 source files = the '115')")
    L.setdefault(pid, ("C", "", "-", "val_family", "a dataset-v1 val problem (protects val, not JAXBench)"))
out, miss = [], []
for pid, r in rows.items():
    lab = L.get(pid)
    if not lab:
        miss.append(pid); continue
    cls, st, task, fam, why = lab
    out.append({"problem_id": pid, "source": r["source"], "rule": " | ".join(r["rules"]), "closest_jaxbench_task": task,
                "class": cls, "b_strength": st or None, "family": fam, "in_kimi_pool_now": pid in KIMI, "justification": why})
print("rows", len(out), "unlabelled", len(miss), miss[:20])
with open(f"{S}/review/exclusions.jsonl", "w") as f:
    for o in out:
        f.write(json.dumps(o) + "\n")
