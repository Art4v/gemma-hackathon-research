"""The full Kimi problem list: every status=ok problem across the pipeline's problem files, deduplicated (problem_id, then ref
sha), minus: the val problems (diag/prompts_val.jsonl), the contamination review's decision=exclude rows (algo_review_kbk.jsonl),
and any task named like a JAXBench task. The user turn is the pipeline's own render_chat.render_problem(ref_code) (render-1), the
same text Claude and the student see."""
import json, pathlib, sys
SC = pathlib.Path("/private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad")
sys.path.insert(0, "/Users/hrudaynara/Projects/gemma-dataset-claude/dataset/scripts")
from render_chat import render_problem  # noqa: E402
SOURCES = ["/Users/hrudaynara/Projects/gemma-dataset-claude/dataset/problems/problems.jsonl",
           f"{SC}/kbk-r1-wt/dataset/problems/problems.jsonl", f"{SC}/kbkc-wt/dataset/problems/problems.jsonl",
           f"{SC}/m1-r1-wt/dataset/problems/problems.jsonl", "/Users/hrudaynara/Projects/gemma-kbk/dataset/problems/problems.jsonl"]
val = {json.loads(l)["task"] for l in open(SC / "diag/prompts_val.jsonl")}
excl = {json.loads(l)["problem_id"] for l in open(SC / "kbk-r1-wt/dataset/kernelbook/algo_review_kbk.jsonl") if json.loads(l)["decision"] == "exclude"}
jax = {json.loads(l)["task"] for l in open("/Users/hrudaynara/Projects/gemma-hackathon-research/control/runs/gemma4_e4b-it-qat__modal/bare.jsonl")}
seen_id, seen_sha, out, drop = set(), set(), [], {"status": 0, "dup": 0, "val": 0, "contamination": 0, "jaxbench_name": 0}
for src in SOURCES:
    for l in open(src):
        r = json.loads(l)
        if r.get("status") != "ok" or r.get("exclude_reason"):
            drop["status"] += 1; continue
        if r["problem_id"] in seen_id or r["ref_sha256"] in seen_sha:
            drop["dup"] += 1; continue
        if r["task_name"] in val:
            drop["val"] += 1; continue
        if r["problem_id"] in excl:
            drop["contamination"] += 1; continue
        if r["task_name"] in jax:
            drop["jaxbench_name"] += 1; continue
        seen_id.add(r["problem_id"]); seen_sha.add(r["ref_sha256"])
        out.append({"problem_id": r["problem_id"], "task_name": r["task_name"], "source": r["source"], "family_id": r["family_id"],
                    "ref_sha256": r["ref_sha256"], "problem_file": src, "user": render_problem(r["ref_code"])})
(SC / "kimi_pilot/full_problems.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
import collections
print(len(out), "problems | dropped", drop, "| by source", dict(collections.Counter(x["source"] for x in out)))
