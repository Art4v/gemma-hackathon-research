"""cpu_check.py, N references at a time. Same checks, same child, same record; only the scheduling differs.

  python dataset/scripts/cpu_check_parallel.py [--workers 3] [--max-rss-gb 8] [--timeout 600] [--force] [--only ID,...]

Each check is still its own isolated `python -I` subprocess with the --max-rss-gb kill. Results are folded into problems.jsonl by
this one process under a lock. Run it after port/author/wrap have finished: they append to the same file. Keep workers x max-rss-gb near the machine's RAM (24 GB here: 3 x 8).
"""
from __future__ import annotations

import argparse, concurrent.futures as cf, json, sys, threading

import cpu_check as cc  # noqa: E402  (same directory)
from common import ORACLE, PROBLEMS, Isolated, now, problems_lock, read_jsonl, write_jsonl  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--force", action="store_true"); ap.add_argument("--only", default="")
    ap.add_argument("--max-rss-gb", type=float, default=8.0); ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--python", default=None)
    a = ap.parse_args(argv)
    path = PROBLEMS / "problems.jsonl"
    only = set(a.only.split(",")) if a.only else None
    files = {p: (ORACLE / p).read_text() for p in cc.ORACLE_FILES}
    sha = cc.oracle_sha()
    todo = [r for r in read_jsonl(path) if r["status"] == "ok" and (only is None or r["problem_id"] in only) and (a.force or "cpu_check" not in r)]
    print(f"cpu_check_parallel: {len(todo)} problems, {a.workers} at a time, cap {a.max_rss_gb} GB each", flush=True)
    lock = threading.Lock()

    def one(r: dict) -> None:
        res = Isolated(a.python, timeout=a.timeout, max_rss_gb=a.max_rss_gb).run(["check.py"], files={**files, "check.py": cc.CHILD, "ref.py": r["ref_code"]},
                                                                                   extra_env={"ORACLE_COMMIT": "cpu_check"})
        line = next((l for l in res.stdout.splitlines() if l.startswith("{")), None)
        rec = cc.fold(json.loads(line) if line else None, res.returncode, res.stderr, getattr(res, "peak_rss_gb", None))
        rec.update(oracle_sha=sha, ref_sha256=r["ref_sha256"], checked_at=now())
        with lock, problems_lock():
            rows = read_jsonl(path)
            for row in rows:
                if row["problem_id"] == r["problem_id"] and row["ref_sha256"] == r["ref_sha256"]:
                    row["cpu_check"] = rec
            write_jsonl(path, rows)
        tag = "PASS" if rec["passed"] else "FAIL " + ",".join(rec["failed_checks"])
        print(f"  {r['problem_id']}: {tag}  run {rec.get('run_s')}s  peak {rec['peak_rss_gb']} GB  {rec.get('error', '')[:120]}", flush=True)

    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, todo))
    checked = [r for r in read_jsonl(path) if r["status"] == "ok" and "cpu_check" in r]
    print(f"passed {sum(r['cpu_check']['passed'] for r in checked)} of {len(checked)} checked")


if __name__ == "__main__":
    main()
