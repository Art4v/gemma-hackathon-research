"""Stage 6b: ingest one oracle job's results, check them against what we submitted, write verdict records and update trajectories.

  python dataset/scripts/oracle_collect.py JOB_ID [--root dataset/] [--jobs-dir oracle/jobs] [--results-dir oracle/results] [--expect-pass]

Reads  <root>/submissions/<JOB_ID>.items.jsonl         what we sent: {id, task, ref_code, code, model}
       <results-dir>/<JOB_ID>/results/*.jsonl          the oracle's rows (or <results-dir>/<JOB_ID>/*.jsonl); meta lines skipped
       <jobs-dir>/<JOB_ID>/MANIFEST.json               if present, oracle/collect.py's own check must say valid
       <root>/candidates/<batch>/r*.jsonl              candidate metadata (problem_id, attempt, sample, code_sha256)
Writes <root>/verdicts/<JOB_ID>.jsonl                  one verdict record per row (PLAN.md schema)
       <root>/trajectories/<batch>.jsonl               one record per problem, attempts merged in; idempotent per job

Fails (exit 1) when: an id is missing, duplicated or unexpected; rows carry more than one oracle_sha; a row's seeds are not
[0,1,2] or its speed_bar is not 0.7; a row's ref_sha256 differs from the reference we submitted; a verdict is unknown; or, with
--expect-pass, any verdict is not pass. Warns when seeds_distinct is false for a task (its create_inputs does not vary by seed).
"""
from __future__ import annotations

import argparse, collections, pathlib, re, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402
from common import (HARNESS_FAULT, N_SEEDS, SPEED_BAR, VERDICTS_ALL, now, read_json, read_jsonl, sha256,  # noqa: E402
                    write_jsonl)

_JOB = re.compile(r"^dataset-(?P<batch>.+)-r(?P<round>\d+)$")
_HUMAN = re.compile(r"\.human\.s\d+$")


def batch_of(job_id: str) -> str:
    m = _JOB.match(job_id)
    return m.group("batch") if m else job_id


def is_human(candidate_id: str, cand: dict | None) -> bool:
    return bool(_HUMAN.search(candidate_id)) or bool(cand and (cand.get("constructor") or {}).get("backend") == "human")


def load_rows(results_dir: pathlib.Path, job_id: str) -> list:
    d = results_dir / job_id
    files = sorted((d / "results").glob("*.jsonl")) or sorted(d.glob("*.jsonl"))
    rows = [r for p in files for r in read_jsonl(p)]
    return [r for r in rows if not r.get("meta")]


def load_candidates(root: pathlib.Path, batch: str) -> dict:
    d = root / "candidates" / batch
    files = sorted(d.glob("r*.jsonl")) if d.exists() else sorted((root / "candidates").glob("**/*.jsonl"))
    out = {}
    for p in files:
        for c in read_jsonl(p):
            out[c["candidate_id"]] = c
    return out


def check_rows(items: list, rows: list, expect_pass: bool) -> tuple:
    """(problems, warnings) for a job's rows against its submitted items."""
    problems, warnings = [], []
    want = {i["id"]: i for i in items}
    count = collections.Counter(r.get("id") for r in rows)
    missing = [i for i in want if not count[i]]
    dup = sorted(i for i, k in count.items() if k > 1 and i in want)
    unexpected = sorted(str(i) for i in count if i not in want)
    if missing:
        problems.append(f"missing rows for {len(missing)} ids: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if dup:
        problems.append(f"duplicate rows for ids: {dup[:5]}")
    if unexpected:
        problems.append(f"unexpected ids in results: {unexpected[:5]}")
    shas = sorted({r.get("oracle_sha") for r in rows})
    if len(shas) > 1:
        problems.append(f"rows carry more than one oracle_sha: {shas}")
    for r in rows:
        i = r.get("id")
        if r.get("seeds") != list(range(N_SEEDS)):
            problems.append(f"{i}: seeds {r.get('seeds')} != {list(range(N_SEEDS))}")
        if r.get("speed_bar") != SPEED_BAR:
            problems.append(f"{i}: speed_bar {r.get('speed_bar')} != {SPEED_BAR}")
        if i in want and r.get("ref_sha256") is not None and r["ref_sha256"] != sha256(want[i]["ref_code"]):
            problems.append(f"{i}: ref_sha256 differs from the submitted reference")
        if r.get("verdict") not in VERDICTS_ALL:
            problems.append(f"{i}: unknown verdict {r.get('verdict')!r}")
        if expect_pass and r.get("verdict") != "pass":
            problems.append(f"{i}: expected pass, got {r.get('verdict')} ({str(r.get('feedback'))[:80]})")
    not_distinct = sorted({r.get("task") for r in rows if r.get("seeds_distinct") is False})
    if not_distinct:
        warnings.append(f"seeds_distinct is false for tasks: {not_distinct} (create_inputs does not vary across seeds)")
    return problems, warnings


def verdict_record(row: dict, job_id: str) -> dict:
    return {"candidate_id": row["id"], "job_id": job_id, "oracle_commit": str(row.get("oracle_commit", "unknown")),
            "oracle_sha": str(row.get("oracle_sha", "unknown")), "verdict": row["verdict"], "feedback": row.get("feedback", ""),
            "speed_vs_xla": row.get("speed_vs_xla"), "seeds_distinct": row.get("seeds_distinct"),
            "accepted": row["verdict"] == "pass", "row": row, "collected_at": now()}


def status_of(rec: dict) -> str:
    attempts = rec.get("attempts", [])
    if any(a["verdict"] == "pass" for a in attempts):
        return "solved"
    if len({a["attempt"] for a in attempts}) >= 3:
        return "unsolved"
    for h in rec.get("harness_events", []):
        superseded = any(a["attempt"] > h["attempt"] or (a["attempt"] == h["attempt"] and a["sample"] > h["sample"]) for a in attempts)
        if not superseded:
            return "harness_blocked"
    return "open"


def merge_trajectories(existing: list, verdicts: list, candidates: dict, items: dict) -> list:
    """Merge this job's verdicts into the per-problem trajectory records. Idempotent: an attempt is keyed by candidate_id."""
    by_problem = {t["problem_id"]: t for t in existing}
    for v in verdicts:
        cid = v["candidate_id"]
        cand = candidates.get(cid)
        if cand is None:
            raise KeyError(f"no candidate record for {cid}; collect needs candidates/<batch>/r*.jsonl")
        rec = by_problem.setdefault(cand["problem_id"], {"problem_id": cand["problem_id"], "task_name": cand["task_name"],
                                                           "attempts": [], "harness_events": [], "human": None, "status": "open"})
        entry = {"attempt": cand["attempt"], "sample": cand["sample"], "candidate_id": cid, "code_sha256": cand["code_sha256"],
                 "job_id": v["job_id"], "verdict": v["verdict"], "feedback": v["feedback"], "speed_vs_xla": v["speed_vs_xla"],
                 "oracle_sha": v["oracle_sha"], "oracle_commit": v["oracle_commit"]}
        if is_human(cid, cand):
            rec["human"] = entry
            continue
        key = "harness_events" if v["verdict"] in HARNESS_FAULT else "attempts"
        for k in ("attempts", "harness_events"):
            rec[k] = [a for a in rec[k] if a["candidate_id"] != cid]
        rec[key].append(entry)
        rec["attempts"].sort(key=lambda a: (a["attempt"], a["sample"]))
        rec["harness_events"].sort(key=lambda a: (a["attempt"], a["sample"]))
    for rec in by_problem.values():
        rec["status"] = status_of(rec)
    return sorted(by_problem.values(), key=lambda t: t["problem_id"])


def collect(job_id: str, root: pathlib.Path, jobs_dir: pathlib.Path, results_dir: pathlib.Path, expect_pass: bool = False) -> int:
    items = read_jsonl(root / "submissions" / f"{job_id}.items.jsonl")
    if not items:
        print(f"no submission at {root / 'submissions' / (job_id + '.items.jsonl')}", file=sys.stderr)
        return 1
    rows = load_rows(results_dir, job_id)
    if not rows:
        print(f"no result rows under {results_dir / job_id}", file=sys.stderr)
        return 1
    problems = []
    job_dir = jobs_dir / job_id
    if (job_dir / "MANIFEST.json").exists():
        sys.path.insert(0, str(common.ORACLE))
        import collect as oracle_collect  # noqa: E402  (oracle/collect.py)
        s = oracle_collect.check(job_dir)
        if not s["valid"]:
            problems.append(f"oracle/collect.py says the job is not valid: " +
                            ", ".join(f"{k}={s[k]}" for k in ("missing", "duplicate", "unexpected", "oracle_sha_mismatch", "seeds_mismatch") if s[k]))
    p, warnings = check_rows(items, rows, expect_pass)
    problems += p
    for w in warnings:
        print(f"warning: {w}")
    if problems:
        print(f"{job_id}: {len(problems)} problem(s)", file=sys.stderr)
        for x in problems:
            print(f"  - {x}", file=sys.stderr)
        return 1
    verdicts = [verdict_record(r, job_id) for r in rows]
    bad = [(v["candidate_id"], e) for v in verdicts for e in common.validate("verdict", v)]
    if bad:
        print(f"{job_id}: verdict records fail the schema: {bad[:5]}", file=sys.stderr)
        return 1
    write_jsonl(root / "verdicts" / f"{job_id}.jsonl", verdicts)
    batch = batch_of(job_id)
    candidates = load_candidates(root, batch)
    traj_path = root / "trajectories" / f"{batch}.jsonl"
    merged = merge_trajectories(read_jsonl(traj_path), verdicts, candidates, {i["id"]: i for i in items})
    write_jsonl(traj_path, merged)
    counts = collections.Counter(v["verdict"] for v in verdicts)
    status = collections.Counter(t["status"] for t in merged)
    print(f"{job_id}: {len(verdicts)} verdicts {dict(sorted(counts.items()))}; trajectories/{batch}.jsonl: {dict(sorted(status.items()))}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("job_id")
    ap.add_argument("--root", default=str(common.DATASET), help="the dataset directory (default: dataset/)")
    ap.add_argument("--jobs-dir", default=None, help="default: <repo>/oracle/jobs")
    ap.add_argument("--results-dir", default=None, help="default: <repo>/oracle/results")
    ap.add_argument("--expect-pass", action="store_true", help="fail unless every verdict is pass (the reverify job)")
    a = ap.parse_args(argv)
    root = pathlib.Path(a.root).resolve()
    jobs = pathlib.Path(a.jobs_dir) if a.jobs_dir else common.ORACLE / "jobs"
    results = pathlib.Path(a.results_dir) if a.results_dir else common.ORACLE / "results"
    return collect(a.job_id, root, jobs, results, a.expect_pass)


if __name__ == "__main__":
    sys.exit(main())
