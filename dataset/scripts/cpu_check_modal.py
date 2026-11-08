"""cpu_check.py on Modal: the same child, oracle files, timeout, memory cap and record, one container per reference, all in parallel.

  modal run dataset/scripts/cpu_check_modal.py [--limit N]      (from the repo root; needs a Modal login)

Why: on the 24 GB build Mac the full-size references pushed it into swap (Nov 8: 24.7 GB swapped, ~1.3 min per check, ~5 h for
232). Each container runs common.Isolated exactly as cpu_check.py does locally (python -I, JAX_PLATFORMS=cpu, OMP_NUM_THREADS=4,
--max-rss-gb 8 kill, 600 s timeout), and cpu_check.fold builds the record. This process is the only writer of problems.jsonl:
results are folded in as they arrive, so stop any local cpu_check first. Records gain `ran_on: modal`.
"""
from __future__ import annotations

import json, pathlib, sys

import modal

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PINS = ["jax==0.10.2", "jaxlib==0.10.2", "numpy==2.5.4", "ml-dtypes==0.6.0", "opt-einsum==3.4.0", "scipy==1.18.1", "jaxtyping==0.3.11", "pydantic==2.14.3"]
image = (modal.Image.debian_slim(python_version="3.12").apt_install("procps").pip_install(*PINS)
         .add_local_dir(str(HERE), "/app/scripts").add_local_dir(str(ROOT / "oracle"), "/app/oracle"))
app = modal.App("dataset-cpu-check", image=image)


@app.function(cpu=2, memory=10240, timeout=900, max_containers=48, retries=1)
def check(problem_id: str, ref_code: str) -> dict:
    sys.path.insert(0, "/app/scripts")
    import cpu_check as cc
    from common import Isolated
    files = {p: pathlib.Path("/app/oracle", p).read_text() for p in cc.ORACLE_FILES}
    res = Isolated(sys.executable, timeout=600, max_rss_gb=8.0).run(["check.py"], files={**files, "check.py": cc.CHILD, "ref.py": ref_code},
                                                                     extra_env={"ORACLE_COMMIT": "cpu_check"})
    line = next((l for l in res.stdout.splitlines() if l.startswith("{")), None)
    return {"problem_id": problem_id, "rec": cc.fold(json.loads(line) if line else None, res.returncode, res.stderr, getattr(res, "peak_rss_gb", None))}


@app.local_entrypoint()
def main(limit: int = 0) -> None:
    sys.path.insert(0, str(HERE))
    import cpu_check as cc
    from common import PROBLEMS, now, read_jsonl, write_jsonl
    path = PROBLEMS / "problems.jsonl"
    todo = [r for r in read_jsonl(path) if r["status"] == "ok" and "cpu_check" not in r]
    todo = todo[:limit] if limit else todo
    sha = cc.oracle_sha()
    print(f"cpu_check_modal: {len(todo)} problems, oracle {sha}", flush=True)
    by_id = {r["problem_id"]: r for r in todo}
    n = 0
    for out in check.starmap([(r["problem_id"], r["ref_code"]) for r in todo], order_outputs=False, return_exceptions=True):
        if isinstance(out, Exception):
            print(f"  container error: {str(out)[:200]}", flush=True); continue
        r, rec = by_id[out["problem_id"]], out["rec"]
        rec.update(oracle_sha=sha, ref_sha256=r["ref_sha256"], checked_at=now(), ran_on="modal")
        rows = read_jsonl(path)
        for row in rows:
            if row["problem_id"] == r["problem_id"] and row["ref_sha256"] == r["ref_sha256"]:
                row["cpu_check"] = rec
        write_jsonl(path, rows)
        n += 1
        tag = "PASS" if rec["passed"] else "FAIL " + ",".join(rec["failed_checks"])
        print(f"  [{n}/{len(todo)}] {r['problem_id']}: {tag}  run {rec.get('run_s')}s  peak {rec['peak_rss_gb']} GB  {rec.get('error', '')[:100]}", flush=True)
    checked = [r for r in read_jsonl(path) if r["status"] == "ok" and "cpu_check" in r]
    print(f"passed {sum(r['cpu_check']['passed'] for r in checked)} of {len(checked)} checked")
