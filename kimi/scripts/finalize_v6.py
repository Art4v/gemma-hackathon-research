"""Stage the final v6 training set and everything behind it into the hruday/kimi-k31 worktree (wt-kimi). Run after assemble_v6.py.
Does not commit or push.
  python3 finalize_v6.py  ->  wt-kimi/kimi/v6/v6_train.jsonl (+ .sha256), v6_stats.json, flag lists, data/new, data/out/rep2, tpu verdicts"""
import collections, glob, hashlib, json, pathlib, shutil
P = pathlib.Path(__file__).resolve().parent
SC = P.parent
W = SC / "wt-kimi/kimi"
J = SC / "tpu/ops1/repo/oracle/jobs"


def cp(src, dst):
    dst = pathlib.Path(dst); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)


# 1. the training set
rows = [json.loads(l) for l in open(P / "v6/v6_train_draft.jsonl")]
ids = [r["example_id"] for r in rows]
assert len(ids) == len(set(ids)), "duplicate example_id"
L = lambda f: {l.strip() for l in open(P / "v6" / f) if l.strip() and not l.startswith("#")}
HARD, UNSURE = L("exclude_hard.txt"), L("flag_unsure.txt")
bad = [r["problem_id"] for r in rows if r["problem_id"] in HARD or r["problem_id"] in UNSURE
       or (r.get("scaled_from") and (r["scaled_from"] in HARD or r["scaled_from"] in UNSURE))]
assert not bad, f"excluded problems in v6: {bad[:5]}"
assert all(r["verdict"] in ("pass", "correct_slow") for r in rows)
import re  # val protection: no dataset-v1 val problem and no copy of one (__bf16 / __scale / ... suffixes) may be trained on
_val = [json.loads(l) for l in open(SC / "diag/prompts_val.jsonl")]
_stems, _vtasks = {re.sub(r"(__[a-z0-9_]+)$", "", v["problem_id"]) for v in _val}, {v["task"] for v in _val}
leak = [r["problem_id"] for r in rows if r["task"] in _vtasks or re.sub(r"(__[a-z0-9_]+)+$", "", r["problem_id"]) in _stems
        or any(r["problem_id"].startswith(s + "__") for s in _stems)]
assert not leak, f"val-family rows in v6: {leak[:5]}"
assert all(r["messages"][-1]["role"] == "assistant" and r["messages"][-1]["content"].startswith("```python\n") for r in rows)
body = "".join(json.dumps(r) + "\n" for r in rows)
(W / "v6").mkdir(parents=True, exist_ok=True)
(W / "v6/v6_train.jsonl").write_text(body)
sha = hashlib.sha256(body.encode()).hexdigest()
(W / "v6/v6_train.jsonl.sha256").write_text(f"{sha}  v6_train.jsonl\n")
# strict fallback: without the rows whose admission depends on A6 (e1 re-admits, rule-3 families), for training before sign-off
dep = [r for r in rows if r["set"] == "e1_readmit" or r["rule3_family"]]
strict = [r for r in rows if not (r["set"] == "e1_readmit" or r["rule3_family"])]
sbody = "".join(json.dumps(r) + "\n" for r in strict)
(W / "v6/v6_train_strict.jsonl").write_text(sbody)
ssha = hashlib.sha256(sbody.encode()).hexdigest()
(W / "v6/v6_train_strict.jsonl.sha256").write_text(f"{ssha}  v6_train_strict.jsonl\n")
(W / "v6/a6_dependent_problem_ids.txt").write_text("# rows whose admission depends on amendment A6 (e1 re-admits + rule-3 families); v6_train_strict.jsonl drops them\n"
                                                   + "".join(sorted({r["problem_id"] + "\n" for r in dep})))
stats = json.load(open(P / "v6/v6_stats.json"))
stats["strict"] = {"file": "kimi/v6/v6_train_strict.jsonl", "sha256": ssha, "rows": len(strict), "a6_dependent_rows": len(dep)}
stats.update({"file": "kimi/v6/v6_train.jsonl", "sha256": sha, "rows": len(rows),
              "by_kind": dict(collections.Counter(r["kind"] for r in rows)),
              "by_set": dict(collections.Counter(r["set"] for r in rows)),
              "flags": {"generic_stack": sum(r["generic_stack"] for r in rows), "rule3_family": sum(r["rule3_family"] for r in rows)},
              "problems": len({r["problem_id"] for r in rows})})
(W / "v6/v6_stats.json").write_text(json.dumps(stats, indent=1))
for f in ("exclude_hard.txt", "flag_unsure.txt", "rule3_families.txt", "flag_generic_stacks.txt", "flag_generic_stacks_new.txt"):
    cp(P / "v6" / f, W / "v6" / f)
old = W / "v6/v6_train_draft.jsonl"
if old.exists():
    old.unlink()  # superseded by v6_train.jsonl
# 2. new problems + their Kimi outputs (first attempts, repairs), CPU scores, logs, ledgers
for f in ("new_problems.jsonl", "new_refs.jsonl", "screen.json", "scan_new.jsonl"):
    cp(P / "new" / f, W / "data/new" / f)
for f in ("scale_problems.jsonl", "scale_refs.jsonl", "scale_log.jsonl", "dropped_after_rescan.json", "scan_scale.jsonl"):
    cp(P / "scale" / f, W / "data/scale" / f)
for f in glob.glob(str(P / "out/new/*")):
    if pathlib.Path(f).is_file():
        cp(f, W / "data/out/new" / pathlib.Path(f).name)
for f in glob.glob(str(P / "out/new/rep2/*")):
    cp(f, W / "data/out/new/rep2" / pathlib.Path(f).name)
for f in glob.glob(str(P / "out/rep2/*")):
    cp(f, W / "data/out/rep2" / pathlib.Path(f).name)
# 3. every TPU verdict behind the dataset (first attempts, repairs, both pools)
ver = []
for d in sorted(glob.glob(str(J / "kimi-*"))):
    for f in sorted(glob.glob(d + "/results/*.jsonl")):
        for l in open(f):
            if not l.strip():
                continue
            x = json.loads(l)
            if x.get("meta"):
                continue
            ver.append({"id": x["id"], "problem_id": x["id"].split("/", 1)[1], "task": x.get("task"), "verdict": x.get("verdict"),
                        "speed_vs_xla": x.get("speed_vs_xla"), "feedback": x.get("feedback"), "failed_stage": x.get("failed_stage"),
                        "oracle_sha": x.get("oracle_sha"), "device_kind": x.get("device_kind"), "job": pathlib.Path(d).name,
                        "halt_victim": x.get("verdict") == "harness_oom" and any(h in (x.get("feedback") or "") for h in ("halted unexpectedly", "RuntimeUnexpectedCoreHalt"))})
(W / "tpu").mkdir(exist_ok=True)
(W / "tpu/tpu_verdicts.jsonl").write_text("".join(json.dumps(v) + "\n" for v in ver))
for f in glob.glob(str(SC / "tpu/ops1/kimi-*.crashes.json")):
    cp(f, W / "tpu" / pathlib.Path(f).name)
# 4. scripts
for f in ("assemble_v6.py", "build_new.py", "run_full.py", "run_repair.py", "run_repair_n.py", "score_cpu.py", "bedrock_kimi.py",
          "finalize_v6.py", "scale/scale_one.py", "scale/build_scale.py"):
    cp(P / f, W / "scripts" / pathlib.Path(f).name)
for f in ("build_items_n.py", "fleet_run_n.sh", "pipeline_n.sh", "fleet_run_a2.sh", "build_rep_items.py", "build_mopup_rep.py", "a2mop_launcher.sh"):
    if (SC / "tpu/fleet" / f).exists():
        cp(SC / "tpu/fleet" / f, W / "tpu" / f)
cp(SC / "tpu/ops1/crashloop.py", W / "tpu/crashloop.py")
print(json.dumps({k: stats[k] for k in ("rows", "sha256", "by_kind", "by_set", "flags", "problems", "strict")}, indent=1))
print("verdict rows:", len(ver), dict(collections.Counter(v["job"].split("-s")[0] for v in ver)))
