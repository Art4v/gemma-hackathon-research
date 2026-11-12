"""A6 rule 2, conservative (Aarav's test, aarav/dataset-v4 @ ad6b273): op_chain minus add/sub/mul/div/neg and layout ops
(convert, broadcast, reshape, transpose, squeeze, expand_dims, copy), collapsed, equals a k-task's chain read the same way -> drop.
Writes a6r2/drops.json {problem_id: [k-tasks]}."""
import json, pathlib, re, glob
S = "/private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad/aarav_ds"
exam = json.load(open(f"{S}/dataset/problems/exam_signatures.json"))
LIN = {"add", "add_any", "sub", "mul", "div", "neg", "copy"}
LAYOUT = ("convert", "broadcast", "reshape", "transpose", "squeeze", "expand_dims", "copy")
def strip(ch):
    out = []
    for o in ch:
        if o in LIN or o.startswith(LAYOUT): continue
        if not out or out[-1] != o: out.append(o)
    return tuple(out)
kt = {}
for t, s in exam.items():
    if re.match(r"^\d+k_", t) and "op_chain" in s:
        kt.setdefault(strip(s["op_chain"]), []).append(t)
print(len([t for t in exam if re.match(r"^\d+k_", t)]), "k-tasks;", len(kt), "distinct stripped chains")
rows = [json.loads(l) for f in sorted(glob.glob("a6r2/out*.jsonl")) for l in open(f)]
err = [r["key"] for r in rows if "error" in r]
drops = {r["key"]: kt[strip(r["op_chain"])] for r in rows if "op_chain" in r and strip(r["op_chain"]) in kt}
json.dump({"rule": __doc__, "errors_unsure_out": err, "drops": drops}, open("a6r2/drops.json", "w"), indent=1)
print(len(rows), "traced,", len(err), "errors (unsure -> out),", len(drops), "matches")
import collections
print(collections.Counter(t for v in drops.values() for t in v).most_common())
r3 = [k for k in drops if "__kn" not in k]; print("scaled matches", len(drops) - len(r3), "| round-3 matches", len(r3))
