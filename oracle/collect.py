"""Check a job's returned results against its MANIFEST.json and summarize them.

  python oracle/collect.py JOB_DIR      -> JOB_DIR/summary.json; exit 1 unless valid

Reads every JOB_DIR/results/*.jsonl. Valid means: every manifest item has exactly one row (no missing, duplicate or unexpected ids;
a --sensitivity job writes one row per floating input under the task's id, so there several rows per id are expected),
and every row and meta line carries the manifest's oracle_sha and seeds. The summary counts rows per verdict and gives the median
speed_vs_xla of the correct kernels (verdict pass or correct_slow). Pure: no JAX.
"""
import collections, json, pathlib, statistics, sys


def check(job: pathlib.Path) -> dict:
    job = pathlib.Path(job)
    m = json.loads((job / "MANIFEST.json").read_text())
    lines = [json.loads(l) for p in sorted((job / "results").glob("*.jsonl")) for l in p.read_text().splitlines() if l.strip()]
    metas, rows = [r for r in lines if r.get("meta")], [r for r in lines if not r.get("meta")]
    want, n = [i["id"] for i in m["items"]], collections.Counter(r.get("id") for r in rows)
    bad = lambda key: sorted({"meta" if r.get("meta") else str(r.get("id")) for r in metas + rows if r.get(key) != m[key]})
    s = {"job_id": m["job_id"], "oracle_sha": m["oracle_sha"], "n_items": len(want), "n_rows": len(rows),
         "missing": [i for i in want if not n[i]], "duplicate": [] if m.get("sensitivity") else sorted(i for i, k in n.items() if k > 1 and i in set(want)),
         "unexpected": sorted(str(i) for i in n if i not in set(want)), "oracle_sha_mismatch": bad("oracle_sha"),
         "seeds_mismatch": bad("seeds")}
    s["valid"] = bool(rows) and not any(s[k] for k in ("missing", "duplicate", "unexpected", "oracle_sha_mismatch", "seeds_mismatch"))
    speeds = [r["speed_vs_xla"] for r in rows if r.get("verdict") in ("pass", "correct_slow") and r.get("speed_vs_xla") is not None]
    s.update(verdicts=dict(sorted(collections.Counter(r.get("verdict") for r in rows).items())), n_correct=len(speeds),
             median_speed_vs_xla_correct=statistics.median(speeds) if speeds else None)
    return s


def main(argv=None) -> None:
    job = pathlib.Path((argv or sys.argv[1:])[0])
    s = check(job)
    (job / "summary.json").write_text(json.dumps(s, indent=1) + "\n")
    print(json.dumps(s, indent=1))
    sys.exit(0 if s["valid"] else 1)


if __name__ == "__main__":
    main()
