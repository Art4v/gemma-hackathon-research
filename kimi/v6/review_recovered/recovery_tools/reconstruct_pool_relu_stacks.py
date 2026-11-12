"""Rebuild review/adv/pool_relu_stacks.json from its complete printout in the main session transcript.

Main session 2c5e8742-...jsonl line 20876 (Bash) loads review/adv/pool_contamination.json and prints, per key,
`list(v.items())[:2]`; the relu-stacks value has only 2 keys, so line 20877 shows it in full (Python repr).
pool_contamination.json's third key is json.load('pool_relu_stacks.json') (ADV L348), so this is that file's content.
It is written with the same call ADV L310 used: json.dump(res, open('pool_relu_stacks.json','w'), indent=1).
Run: python3 -I reconstruct_pool_relu_stacks.py
"""
import ast, json, os

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = ("/Users/hrudaynara/.claude/projects/-Users-hrudaynara-Projects-gemma-hackathon-research/"
        "2c5e8742-e04c-4af6-a34a-5943a12b6e62.jsonl")
ADV_DIR_OLD = ("/private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/"
               "2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad/review/adv")
log, fails = [], []
def check(cond, msg):
    s = ("PASS " if cond else "FAIL ") + msg; print(s); log.append(s)
    if not cond: fails.append(msg)

msgs = {}
with open(MAIN, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if i in (20876, 20877, 21653, 21654):
            msgs[i] = json.loads(line)["message"]["content"]
        if i > 21654: break
use = [p for p in msgs[20876] if p.get("type") == "tool_use"][0]
res = [p for p in msgs[20877] if p.get("type") == "tool_result"][0]
check(res["tool_use_id"] == use["id"] and f'A={ADV_DIR_OLD};' in use["input"]["command"]
      and 'pc = json.load(open("$A/pool_contamination.json"))' in use["input"]["command"],
      f"main L20876 ({use['id']}) reads review/adv/pool_contamination.json; L20877 is its result")
text = res["content"] if isinstance(res["content"], str) else "".join(x.get("text", "") for x in res["content"])
lines = {ln.split(" ", 1)[0]: ln for ln in text.split("\n") if ln}
key3 = lines["relu_stacks_meeting_audit_A_standard"]
pre = "relu_stacks_meeting_audit_A_standard dict "
check(key3.startswith(pre) and key3.endswith("]])]"), "relu-stacks line is complete (ends with ']])]')")
items = ast.literal_eval(key3[len(pre):])
check(len(items) == 2 and lines["relu:"] == "relu: dict " + repr([k for k, _ in items]),
      "both keys printed, same keys/order as pool_relu_stacks.json (L20877 'relu:' line)")
stacks = {k: v for k, v in items}
for k, v in stacks.items():
    print(k, len(v), "| in audit table:", sum(1 for x in v if x[2] != "-"), "| share>=0.95:", sum(1 for x in v if x[1] >= 0.95))
a, b = list(stacks.values())
check((len(a), sum(x[2] != "-" for x in a), sum(x[1] >= 0.95 for x in a)) == (91, 2, 52)
      and (len(b), sum(x[2] != "-" for x in b), sum(x[1] >= 0.95 for x in b)) == (68, 3, 49),
      "counts == ADV L311 (91 | 2 | 52 and 68 | 3 | 49)")
check(all(isinstance(x[0], str) and isinstance(x[1], float) and isinstance(x[2], str) and len(x) == 3 for v in stacks.values() for x in v),
      "every entry is [problem_id:str, share:float, audit_class:str]")
with open(OUT + "/regen/review/adv/pool_relu_stacks.json", "w") as fh:
    json.dump(stacks, fh, indent=1)          # identical call to ADV L310
check(os.path.getsize(OUT + "/regen/review/adv/pool_relu_stacks.json") == 9557, "pool_relu_stacks.json size == 9557 (ADV L357 ls)")

# keys 1-2 of the L20877 printout (first two items each) must agree with the L348 literals
for k in ("verified_instance_in_pool_not_in_audit", "unsure_in_pool_not_in_audit"):
    log.append(f"L20877 {k}: " + lines[k][:120])
with open(OUT + "/checks/l20877_keys12.txt", "w") as fh:
    fh.write(lines["verified_instance_in_pool_not_in_audit"] + "\n" + lines["unsure_in_pool_not_in_audit"] + "\n")
r2 = [p for p in msgs[21654] if p.get("type") == "tool_result"][0]
t2 = r2["content"] if isinstance(r2["content"], str) else "".join(x.get("text", "") for x in r2["content"])
with open(OUT + "/checks/l21654_prefixes.txt", "w") as fh:
    fh.write("\n".join(ln for ln in t2.split("\n") if ln.split(" ", 1)[0] in (
        "verified_instance_in_pool_not_in_audit", "unsure_in_pool_not_in_audit", "relu_stacks_meeting_audit_A_standard")) + "\n")
with open(OUT + "/checks/reconstruct_pool_relu_stacks.log", "w") as fh:
    fh.write("\n".join(log) + "\n")
print("FAILURES:", fails)
