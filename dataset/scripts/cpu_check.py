"""Oracle-readiness of every reference, on CPU at full CONFIG size. A problem enters the dataset only if every check passes.

  python dataset/scripts/cpu_check.py [--force] [--only PROBLEM_ID,...] [--max-rss-gb 8] [--timeout 900]

Each reference runs alone in an isolated `python -I` subprocess with the oracle's own files copied in (tpu_oracle.py, tpu_check.py,
outside_ops.py), so inputs, seeds and the answer key are built exactly as the TPU oracle builds them:

  loads              tpu_oracle.Task(name, code).load(): the module imports, create_inputs shapes resolve, static_argnums honoured
  fits_hbm           3 x input bytes <= tpu_check.HBM_BUDGET_BYTES (28e9); otherwise the oracle never runs the task
  deterministic      two create_inputs() calls for seed 0 are equal leaf by leaf (the oracle refuses the task otherwise)
  seeds_distinct     seeds 0, 1, 2 (the oracle's PRNG fold-in) give three different input fingerprints
  runs               jax.jit(workload) executes on seed 0 at full size
  finite             every floating output leaf is finite
  highest_precision  the oracle's answer key (float32 inputs, matmul precision "highest") also runs; its error vs default recorded

Results are written onto the problem record (`cpu_check`) after each problem, so an interrupted run resumes where it stopped.
Memory: one subprocess at a time, killed above --max-rss-gb (the build machine has 17 GB; see PROCESS.md, Memory).
"""
from __future__ import annotations

import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ORACLE, PROBLEMS, Isolated, now, read_jsonl, sha256, write_jsonl  # noqa: E402

ORACLE_FILES = ("tpu_oracle.py", "tpu_check.py", "outside_ops.py")
CHECKS = ("loads", "fits_hbm", "deterministic", "seeds_distinct", "runs", "finite", "highest_precision")

CHILD = r'''
import json, sys, time
sys.path.insert(0, ".")
import tpu_oracle as o
o.init(cpu_smoke=True)
jax, jnp, np, tc = o.jax, o.jnp, o.np, o.tc
r = {"jax": jax.__version__}
def done(**kw):
    r.update(kw); print(json.dumps(r)); sys.exit(0)
t = o.Task("task", open("ref.py").read())
try:
    t.load()
except Exception as e:
    done(loads=False, error=f"load: {type(e).__name__}: {str(e)[-300:]}")
r.update(loads=True, input_bytes=t.info["full_input_bytes"], shapes=t.info["shapes"], input_dtype=t.info["input_dtype"])
r["fits_hbm"] = 3 * t.info["full_input_bytes"] <= tc.HBM_BUDGET_BYTES
try:
    x0 = t.inputs(0)
    r["deterministic"] = bool(o._same_inputs(x0, t.inputs(0)))
    fps = [o._fingerprint(x0)]
    for s in (1, 2):
        xs = t.inputs(s); fps.append(o._fingerprint(xs)); del xs
    r["seeds_distinct"] = len(set(fps)) == len(fps)
except Exception as e:
    done(error=f"inputs: {type(e).__name__}: {str(e)[-300:]}")
try:
    f = t.jit(t.ref.workload)
    t0 = time.time(); out = o._host(jax.block_until_ready(f(*x0))); r["run_s"] = round(time.time() - t0, 2)
    r["runs"] = True
except Exception as e:
    done(runs=False, error=f"run: {type(e).__name__}: {str(e)[-300:]}")
leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(out)]
r["output_shapes"] = [list(x.shape) for x in leaves]
r["output_dtypes"] = [str(x.dtype) for x in leaves]
r["finite"] = all(bool(np.isfinite(x.astype(np.float32)).all()) for x in leaves if np.issubdtype(x.dtype, np.floating) or "float" in str(x.dtype))
state = {"precision": "f32_highest", "failed": []}
try:
    t0 = time.time(); want = t._key(x0, state); r["highest_s"] = round(time.time() - t0, 2)
    r["highest_precision"] = want is not None
    r["ref_precision"] = state["precision"]
    if want is not None:
        e = tc.rel_err(want, out); r["err_default_vs_highest"] = None if e is None else float(e)
    if state["failed"]:
        r["ref_precision_failed"] = state["failed"]
except Exception as e:
    r["highest_precision"] = False; r["error"] = f"highest: {type(e).__name__}: {str(e)[-300:]}"
done()
'''


def oracle_sha() -> str:
    return sha256(b"".join((ORACLE / p).read_bytes() for p in ORACLE_FILES))[:12]


def fold(child: dict | None, rc: int, stderr: str, peak) -> dict:
    """The cpu_check record from a child's JSON line (or its absence). passed only if every check is True."""
    res = dict(child or {})
    if child is None:
        tail = (stderr.strip().splitlines() or ["no output"])[-1][-300:]
        res["error"] = "memory cap" if rc == 137 else "timeout" if rc == 124 else f"crashed (rc {rc}): {tail}"
        if rc == 137 and "[memory cap" in stderr:
            res["error"] = stderr[stderr.index("[memory cap"):].strip()
    for c in CHECKS:
        res[c] = bool(res.get(c, False))
    res["passed"] = all(res[c] for c in CHECKS)
    res["failed_checks"] = [c for c in CHECKS if not res[c]]
    res["peak_rss_gb"] = peak
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true"); ap.add_argument("--only", default="")
    ap.add_argument("--max-rss-gb", type=float, default=8.0); ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--python", default=None)
    a = ap.parse_args(argv)
    path = PROBLEMS / "problems.jsonl"
    rows = read_jsonl(path)
    only = set(a.only.split(",")) if a.only else None
    files = {p: (ORACLE / p).read_text() for p in ORACLE_FILES}
    sha = oracle_sha()
    todo = [r for r in rows if r["status"] == "ok" and (only is None or r["problem_id"] in only) and (a.force or "cpu_check" not in r)]
    print(f"cpu_check: {len(todo)} problems, one at a time, cap {a.max_rss_gb} GB", flush=True)
    for r in todo:
        res = Isolated(a.python, timeout=a.timeout, max_rss_gb=a.max_rss_gb).run(["check.py"], files={**files, "check.py": CHILD, "ref.py": r["ref_code"]},
                                                                                   extra_env={"ORACLE_COMMIT": "cpu_check"})
        line = next((l for l in res.stdout.splitlines() if l.startswith("{")), None)
        rec = fold(json.loads(line) if line else None, res.returncode, res.stderr, getattr(res, "peak_rss_gb", None))
        rec.update(oracle_sha=sha, ref_sha256=r["ref_sha256"], checked_at=now())
        r["cpu_check"] = rec
        write_jsonl(path, rows)  # after every problem: resumable
        tag = "PASS" if rec["passed"] else "FAIL " + ",".join(rec["failed_checks"])
        print(f"  {r['problem_id']}: {tag}  run {rec.get('run_s')}s  peak {rec['peak_rss_gb']} GB  {rec.get('error', '')[:120]}", flush=True)
    checked = [r for r in rows if r["status"] == "ok" and "cpu_check" in r]
    print(f"passed {sum(r['cpu_check']['passed'] for r in checked)} of {len(checked)} checked")


if __name__ == "__main__":
    main()
