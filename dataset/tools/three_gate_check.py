"""Both free CPU gates for one kernel, in the order the TPU would hit them:
  1. the oracle's own step 1 (tpu_oracle.py --static-only, the call prescreen.py makes): import, workload, the kernel check
  2. TPU v6e lowering on this CPU (curation/lower_filter.py filter): Pallas's TPU lowering, which enforces block-shape rules and
     catches Mosaic errors without a chip. Reject-only: anything it doesn't reject still goes to the chip, which is the only judge of a pass.
  3. numerics in TPU interpret mode on CPU vs the reference (np.allclose rtol=atol=1e-2, seed 0, full size; skipped if >300 s)
  python check.py TASK_NAME KERNEL_FILE   ->  STATIC ... / LOWER ... / NUMERIC ... / then READY (submit) or FIX (read the feedback, edit, re-run)"""
import json, pathlib, subprocess, sys, tempfile
sys.path.insert(0, "/Users/hrudaynara/Projects/gemma-dataset-claude/dataset/scripts")
from prescreen import static_check  # noqa: E402
PY = "/private/tmp/claude-501/-Users-hrudaynara-Projects-gemma-hackathon-research/2c5e8742-e04c-4af6-a34a-5943a12b6e62/scratchpad/lowervenv/bin/python"   # JAX 0.10.2 (the oracle's pin), linked straight to the real interpreter so the lowering sandbox can exec it
PYNUM = "/Users/hrudaynara/Projects/gemma-dataset-gen/dataset/.venv/bin/python"   # numerics child (unsandboxed, like the static step)
LF = "/Users/hrudaynara/Projects/gemma-hackathon-research/curation/lower_filter.py"
task, kfile = sys.argv[1], sys.argv[2]
W = pathlib.Path(__file__).resolve().parent
ref = (W / "problems" / f"{task}.ref.py").read_text()
code = pathlib.Path(kfile).read_text()
rows, _ = static_check([{"id": "k", "task": task, "code": code}], {task: ref}, None, 240)
r = rows.get("k") or next(iter(rows.values()), {"verdict": "harness", "feedback": "no output"})
print("STATIC", r.get("verdict"), "|", (r.get("feedback") or "")[:700])
if r.get("verdict") != "static_ok":
    print("FIX"); sys.exit(1)
with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    (d / "items.jsonl").write_text(json.dumps({"task": task, "ref": ref}) + "\n")
    (d / "rows.jsonl").write_text(json.dumps({"id": "k", "task": task, "code": code}) + "\n")
    p = subprocess.run([sys.executable, LF, "filter", "--py", PY, "--items", str(d / "items.jsonl"), "--rows", str(d / "rows.jsonl"),
                        "--out", str(d / "lower.jsonl"), "--feedback-out", str(d / "fb.jsonl"), "--workers", "1"], capture_output=True, text=True)
    out = [json.loads(l) for l in (d / "lower.jsonl").read_text().splitlines()] if (d / "lower.jsonl").exists() else []
    fb = {json.loads(l)["id"]: json.loads(l)["feedback"] for l in (d / "fb.jsonl").read_text().splitlines()} if (d / "fb.jsonl").exists() else {}
    o = out[0] if out else {"outcome": "abstain", "error": (p.stderr or "no output")[-300:]}
    print("LOWER", o.get("outcome"), "|", (fb.get("k") or o.get("error") or "")[:700])
    if o.get("outcome") == "rejected":
        print("FIX"); sys.exit(1)
# 3. numerics: the kernel in TPU interpret mode on CPU vs the reference in float32 ("highest"), seed-0 inputs at full size,
#    np.allclose(rtol=atol=1e-2) like the oracle's step 3. Slow problems are skipped (the TPU still judges them).
import fcntl, time


def numeric_slot(n=3):
    """At most n numeric checks at once across all agents (interpret mode at full size can use several GB)."""
    while True:
        for i in range(n):
            f = open(W / f".numlock{i}", "w")
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return f
            except BlockingIOError:
                f.close()
        time.sleep(2)


slot = numeric_slot()
with tempfile.TemporaryDirectory() as d:
    try:
        p = subprocess.run([PYNUM, "-I", str(W / "numeric_child.py"), str(W / "problems" / f"{task}.ref.py"),
                            str(pathlib.Path(kfile).resolve())], capture_output=True, text=True, timeout=300, cwd=d)
        line = [l for l in p.stdout.splitlines() if l.startswith("{")]
        if not line:
            err = (p.stderr.strip().splitlines() or ["no output"])[-1][:600]
            print("NUMERIC error |", err); print("FIX"); sys.exit(1)
        n = json.loads(line[-1])
        print("NUMERIC", "ok" if n["ok"] else "WRONG", "|", n.get("detail") or f"max abs err {n['max_abs_err']:.3g} ({n['s']}s)")
        print("READY" if n["ok"] else "FIX"); sys.exit(0 if n["ok"] else 1)
    except subprocess.TimeoutExpired:
        print("NUMERIC skipped | too slow in CPU interpret mode (>300 s); the TPU will judge correctness"); print("READY"); sys.exit(0)
