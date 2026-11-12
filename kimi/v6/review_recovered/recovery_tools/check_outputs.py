"""Compare the replayed steps' stdout with the original tool_result text, and run the count checks.
Run after regen/replay_L*.sh: python3 -I check_outputs.py
"""
import collections, hashlib, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
WF = ("/Users/hrudaynara/.claude/projects/-Users-hrudaynara-Projects-gemma-hackathon-research/"
      "2c5e8742-e04c-4af6-a34a-5943a12b6e62/subagents/workflows/wf_49512077-aae")
ADV = WF + "/agent-a155bcc295ec0cc51.jsonl"
V6 = "/Users/hrudaynara/Projects/gemma-work/wt-kimi/kimi/v6"
RA = OUT + "/regen/review/adv"
log, fails = [], []
def check(cond, msg):
    s = ("PASS " if cond else "FAIL ") + msg; print(s); log.append(s)
    if not cond: fails.append(msg)


def tr(lineno, kind):
    with open(ADV, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i == lineno:
                d = json.loads(line)
                for p in d["message"]["content"]:
                    if p.get("type") == kind == "tool_result":
                        c = p["content"]; return "".join(x.get("text", "") for x in c) if isinstance(c, list) else c
                    if p.get("type") == kind == "tool_use":
                        return p["input"]


# 1. replayed stdout == original tool_result
for n in (286, 290, 314, 352, 360, 364):
    got = open(f"{OUT}/checks/replay_L{n}.out", encoding="utf-8").read()
    want = tr(n + 1, "tool_result")
    check(got.rstrip("\n") == want.rstrip("\n"), f"replay L{n} stdout == transcript L{n + 1} tool_result ({len(want)} chars)")
got = open(f"{OUT}/checks/replay_L356.out", encoding="utf-8").read().split("\n")
want = tr(357, "tool_result").split("\n")
check(got[:8] == want[:8], "replay L356 group counts == transcript L357 (8 lines)")
sizes = dict(reversed(x.split(" ", 1)) for x in want[9:] if " " in x)
for f in ("adversarial_verdicts.jsonl", "final_safe_list.txt", "verdicts.py", "pool_relu_stacks.json", "pool_contamination.json"):
    check(os.path.getsize(f"{RA}/{f}") == int(sizes[f]), f"size of regenerated {f} = {os.path.getsize(f'{RA}/{f}')} == L357 ls {sizes[f]}")
got = open(f"{OUT}/checks/replay_L348_heredoc2.out", encoding="utf-8").read().strip()
check(got == "29 7" and tr(349, "tool_result").rstrip("\n").split("\n")[-1] == got, "replay of L348 heredoc 2 prints '29 7' == last line of transcript L349")
pc = json.load(open(f"{RA}/pool_contamination.json"))
want = {l.split(" ", 1)[0]: l.split(" ", 1)[1] for l in open(f"{OUT}/checks/l21654_prefixes.txt").read().split("\n") if l}
check(list(pc) == list(want) and all(json.dumps(v)[:400] == want[k] for k, v in pc.items()),
      "pool_contamination.json: json.dumps(value)[:400] of all 3 keys == main-session L21654 printout")
w12 = [l for l in open(f"{OUT}/checks/l20877_keys12.txt").read().split("\n") if l]
check(all(l == k + " dict " + repr(list(pc[k].items())[:2]) for l in w12 for k in [l.split(" ", 1)[0]]),
      "pool_contamination.json keys 1-2: first two items == main-session L20877 printout")

# 2. count checks on the regenerated files
V = [json.loads(l) for l in open(f"{RA}/adversarial_verdicts.jsonl")]
c = collections.Counter(v["verdict"] for v in V)
check(len(V) == 273 and c["safe to re-admit"] == 191 and c["keep excluded (instance)"] == 43 and c["keep excluded (unsure, default)"] == 39,
      f"273 rows: 191 safe / 43 keep-excluded instance / 39 unsure ({dict(c)})")
inpool = [v for v in V if v["in_kimi_pool_now"]]
cp = collections.Counter(v["verdict"] for v in inpool)
check(len(inpool) == 87 and cp["keep excluded (instance)"] == 29 and cp["keep excluded (unsure, default)"] == 16 and cp["safe to re-admit"] == 42,
      f"87 class-B in Kimi pool: 45 out (29 copies + 16 unsure), 42 stay ({dict(cp)})")
check(sum(1 for v in V if v["verdict"] == "safe to re-admit" and not v["in_kimi_pool_now"]) == 149, "149 safe problems not yet in the pool")
v4 = [v for v in V if v["source"] == "authored/scraped (dataset-v1) or m1 child"]
check(len(v4) == 52 and sum(1 for v in v4 if v["verdict"] != "safe to re-admit") == 42, "v4 family drop: 42 of its 52 class-B stay out")
so = tr(368, "tool_use")
check(so["details"]["verdict_counts"]["total_class_B_reviewed"] == 273 and so["details"]["verdict_counts"]["already_in_kimi_pool"] == {"safe_keep": 42, "remove_instance": 29, "remove_unsure": 16},
      "StructuredOutput verdict_counts agree (273; in pool 42/29/16)")
fsl = open(f"{RA}/final_safe_list.txt").read()
check(fsl == "\n".join(so["details"]["final_safe_list"]) + "\n", "final_safe_list.txt == StructuredOutput details.final_safe_list (191, same order)")
check(fsl == "\n".join(json.loads(tr(361, "tool_result"))) + "\n", "final_safe_list.txt == the list printed at L361 (json.dumps of the file, untruncated)")
check(sorted(v["problem_id"] for v in V if v["verdict"] == "safe to re-admit") == fsl.split(), "final_safe_list.txt == safe rows of adversarial_verdicts.jsonl")

# 3. every keep-excluded / unsure problem is in v6 exclude_hard.txt U flag_unsure.txt
def ids(f):
    return {l.strip() for l in open(f"{V6}/{f}") if l.strip() and not l.lstrip().startswith("#")}
hard, unsure = ids("exclude_hard.txt"), ids("flag_unsure.txt")
ke = [v for v in V if v["verdict"] != "safe to re-admit"]
miss = [v["problem_id"] for v in ke if v["problem_id"] not in hard | unsure]
check(not miss, f"all 82 keep-excluded/unsure in exclude_hard.txt U flag_unsure.txt (missing {len(miss)}: {miss})")
inst_not_hard = [v["problem_id"] for v in ke if v["verdict"] == "keep excluded (instance)" and v["problem_id"] not in hard]
uns_not_listed = [v["problem_id"] for v in ke if v["verdict"].startswith("keep excluded (unsure") and v["problem_id"] not in unsure]
print("info: instance verdicts not in exclude_hard.txt:", inst_not_hard)
print("info: unsure verdicts not in flag_unsure.txt:", uns_not_listed)
log.append(f"info: instance verdicts not in exclude_hard.txt ({len(inst_not_hard)}): {inst_not_hard}")
log.append(f"info: unsure verdicts not in flag_unsure.txt ({len(uns_not_listed)}): {uns_not_listed}")
safe_flagged = [v["problem_id"] for v in V if v["verdict"] == "safe to re-admit" and v["problem_id"] in hard | unsure]
print("info: safe verdicts that v6 nevertheless excludes/flags:", safe_flagged)
log.append(f"info: safe verdicts that v6 excludes/flags ({len(safe_flagged)}): {safe_flagged}")
dep = ids("a6_dependent_problem_ids.txt")
print("info: a6_dependent_problem_ids.txt:", len(dep), "| in final_safe_list:", len(dep & set(fsl.split())), "| not in it:", sorted(dep - set(fsl.split()))[:20])
log.append(f"info: a6_dependent_problem_ids.txt {len(dep)} ids; in final_safe_list {len(dep & set(fsl.split()))}; not in it: {sorted(dep - set(fsl.split()))}")

for f in ("adversarial_verdicts.jsonl", "final_safe_list.txt", "verdicts.py", "pool_contamination.json", "pool_relu_stacks.json"):
    h = hashlib.sha256(open(f"{RA}/{f}", "rb").read()).hexdigest(); log.append(f"sha256 {f} {h}"); print("sha256", f, h)
with open(f"{OUT}/checks/check_outputs.log", "w") as f:
    f.write("\n".join(log) + "\n")
print("FAILURES:", fails)
