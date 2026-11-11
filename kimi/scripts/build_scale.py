"""e3 scale variants: every pool problem Kimi already solved (in v6/v6_train_draft.jsonl: pass, correct_slow or repaired) whose
inputs are < 200 MB and which carries no contamination flag at all (not hard / unsure / rule-3 family / generic GEMM-conv+ReLU stack),
re-sized by scale_one.py to 200-800 MB inputs (shape-checked with jax.eval_shape on CPU; the reference itself first runs at the
new size on the TPU, inside the oracle) (JAXBench k tasks: median 384 MB). Same op chain, same family_id; problem_id and
task_name get a __scale suffix. Output: scale/scale_refs.jsonl, scale/scale_problems.jsonl (full_problems.jsonl format), scale/scale_log.jsonl"""
import collections, concurrent.futures as cf, json, os, pathlib, subprocess, sys
P = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, "/Users/hrudaynara/Projects/gemma-dataset-claude/dataset/scripts")
from render_chat import render_problem  # noqa: E402
import hashlib
PY = "/Users/hrudaynara/Projects/gemma-dataset-gen/dataset/.venv/bin/python"
probs = {json.loads(l)["problem_id"]: json.loads(l) for l in open(P / "full_problems.jsonl")}
meta = {}
for pf in {p["problem_file"] for p in probs.values()}:
    for l in open(pf):
        r = json.loads(l)
        if r["problem_id"] in probs and r["problem_id"] not in meta:
            meta[r["problem_id"]] = r
L = lambda f: {l.strip() for l in open(P / "v6" / f) if l.strip() and not l.startswith("#")}
FLAG = L("exclude_hard.txt") | L("flag_unsure.txt") | L("rule3_families.txt") | L("flag_generic_stacks.txt")
solved = {json.loads(l)["problem_id"] for l in open(P / "v6/v6_train_draft.jsonl")}
cand = [pid for pid in sorted(solved) if pid not in FLAG and (meta[pid].get("input_bytes") or 0) and meta[pid]["input_bytes"] < 200e6]
print(len(cand), "candidates", flush=True)


def one(pid):
    inp = json.dumps({"code": meta[pid]["ref_code"]})
    for run_cpu in (os.environ.get("RUN_CPU", "0"),):
        try:
            p = subprocess.run([PY, str(P / "scale/scale_one.py")], input=inp, capture_output=True, text=True, timeout=240 if run_cpu == "1" else 120,
                               env={"RUN_CPU": run_cpu, "JAX_PLATFORMS": "cpu", "PATH": "/usr/bin:/bin"})
            line = [l for l in p.stdout.splitlines() if l.startswith("{")]
            d = json.loads(line[-1]) if line else {"ok": False, "reason": "no output: " + p.stderr[-300:]}
            return pid, d
        except subprocess.TimeoutExpired:
            continue
    return pid, {"ok": False, "reason": "timeout"}


refs, out, log, c = [], [], [], collections.Counter()
with cf.ThreadPoolExecutor(int(os.environ.get("WORKERS", "8"))) as ex:
    for n, (pid, d) in enumerate(ex.map(one, cand), 1):
        log.append({"problem_id": pid, **{k: v for k, v in d.items() if k != "code"}})
        c["ok" if d.get("ok") else "fail"] += 1
        if n % 20 == 0:
            print(n, dict(c), flush=True)
        if not d.get("ok"):
            continue
        m = meta[pid]; code = d["code"]
        row = {"problem_id": pid + "__scale", "task_name": m["task_name"] + "__scale", "source": m["source"], "family_id": m["family_id"],
               "ref_sha256": hashlib.sha256(code.encode()).hexdigest(), "set": "e3_scale", "scaled_from": pid}
        refs.append({**row, "ref_code": code, "input_bytes": d["bytes_new"], "scale": {k: d[k] for k in ("key", "old", "new", "factor", "bytes_old", "cpu")}})
        out.append({**row, "problem_file": str(P / "scale/scale_refs.jsonl"), "user": render_problem(code)})
(P / "scale/scale_refs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in refs))
(P / "scale/scale_problems.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
(P / "scale/scale_log.jsonl").write_text("".join(json.dumps(x) + "\n" for x in log))
print("SCALE BUILD DONE", len(out), "variants of", len(cand), dict(c), "| keys", collections.Counter(r["scale"]["key"] for r in refs).most_common(8),
      "| cpu", collections.Counter(r["scale"]["cpu"][:12] for r in refs), flush=True)
