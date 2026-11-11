"""New Kimi problems for v6 (Hruday's go, Nov 11 night):
  e1  the 134 class-B problems the adversarial review judged 'safe to re-admit' under the instance-level rule
      (review/adv/final_safe_list.txt) that are not in the pool and have a reference
  e2  Karthik's kbkd batch (karthik/kbk@036fe89, status ok, 224): it already passed the pipeline's contamination layers
Same filters as build_full.py (dedupe by problem_id and ref sha against the pool and each other, val exact task name, JAXBench task
name) plus the val-family check Aarav's agent ran (a task whose name minus a __bf16/__m1_aux/... suffix is a val task), and the
reference must have passed its CPU checks (finite, fits_hbm). Output: new/new_refs.jsonl (problem rows with ref_code) and
new/new_problems.jsonl (full_problems.jsonl format, user = render_problem(ref_code), render-1)."""
import json, pathlib, re, sys, collections
SC = pathlib.Path("/private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad")
P = SC / "kimi_pilot"
sys.path.insert(0, "/Users/hrudaynara/Projects/gemma-dataset-claude/dataset/scripts")
from render_chat import render_problem  # noqa: E402
pool = {json.loads(l)["problem_id"]: json.loads(l) for l in open(P / "full_problems.jsonl")}
pool_sha = {r["ref_sha256"] for r in pool.values()}
val = [json.loads(l) for l in open(SC / "diag/prompts_val.jsonl")]
val_tasks = {v["task"] for v in val}
val_stems = {re.sub(r"(__[a-z0-9_]+)$", "", v["problem_id"].split("/")[-1]) for v in val}
jax = {json.loads(l)["task"] for l in open("/Users/hrudaynara/Projects/gemma-hackathon-research/control/runs/gemma4_e4b-it-qat__modal/bare.jsonl")}
SRC = [SC / "review/karthik_kbk_problems.jsonl", SC / "review/aarav_kbkc_problems.jsonl",
       pathlib.Path("/Users/hrudaynara/Projects/gemma-dataset-claude/dataset/problems/problems.jsonl"),
       pathlib.Path("/Users/hrudaynara/Projects/gemma-dataset-gen/dataset/problems/problems.jsonl")]
safe = [l.strip() for l in open(SC / "review/adv/final_safe_list.txt") if l.strip()]
want = {p: "e1_readmit" for p in safe if p not in pool}
rows = {}
for src in SRC:
    for l in open(src):
        r = json.loads(l)
        pid = r["problem_id"]
        if pid in want and pid not in rows and r.get("ref_code"):
            rows[pid] = (want[pid], r)
for l in open(SC / "review/karthik_kbk_problems.jsonl"):
    r = json.loads(l)
    if r.get("batch") == "kbkd" and r["status"] == "ok" and r["problem_id"] not in rows:
        rows[r["problem_id"]] = ("e2_kbkd", r)
seen_id, seen_sha, out, refs, drop = set(pool), set(pool_sha), [], [], collections.Counter()
for pid, (tag, r) in rows.items():
    stem = re.sub(r"(__[a-z0-9_]+)$", "", pid.split("/")[-1])
    cc = r.get("cpu_check") or {}
    if pid in seen_id or r["ref_sha256"] in seen_sha:
        drop[tag + ":dup"] += 1; continue
    if r["task_name"] in val_tasks or (pid.split("/")[0] in ("kb", "authored", "scraped") and stem in val_stems):
        drop[tag + ":val"] += 1; continue
    if r["task_name"] in jax:
        drop[tag + ":jaxbench_name"] += 1; continue
    if r.get("fits_hbm") is False or cc.get("finite") is False or (cc.get("failed_checks") or []):
        drop[tag + ":ref_checks"] += 1; continue
    seen_id.add(pid); seen_sha.add(r["ref_sha256"])
    refs.append({"problem_id": pid, "task_name": r["task_name"], "source": r["source"], "family_id": r["family_id"],
                 "ref_sha256": r["ref_sha256"], "ref_code": r["ref_code"], "input_bytes": r.get("input_bytes"), "set": tag})
    out.append({"problem_id": pid, "task_name": r["task_name"], "source": r["source"], "family_id": r["family_id"], "set": tag,
                "ref_sha256": r["ref_sha256"], "problem_file": str(P / "new/new_refs.jsonl"), "user": render_problem(r["ref_code"])})
(P / "new/new_refs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in refs))
(P / "new/new_problems.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
print(len(out), "new problems", dict(collections.Counter(x["set"] for x in out)), "| dropped", dict(drop))
print("not found (no reference):", [p for p in want if p not in rows])
