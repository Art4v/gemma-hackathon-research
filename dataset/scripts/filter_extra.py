"""Contamination layers filter_contamination.py lacks, for one batch: op-multiset Jaccard >= 0.8 against every JAXBench task
(level 1, which filter_contamination only sends to review, and the 14 level-2-only (problems/exam_signatures_level2.json) and the val problems (final/val.jsonl), and shape leakage (same input
shapes and op-multiset Jaccard >= 0.5 as any exam task, level 1 or 2). Run after filter_contamination.py has filled signatures.
  python dataset/scripts/filter_extra.py --batch kbk [--dry-run]
"""
from __future__ import annotations

import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import DATASET, PROBLEMS, now, read_jsonl, write_jsonl  # noqa: E402
from signature import jaccard  # noqa: E402

BAR, SHAPE_BAR = 0.8, 0.5


def banned() -> dict:
    out = {f"exam:{k}": v for k, v in json.load(open(PROBLEMS / "exam_signatures.json")).items()}
    out.update({f"exam:{k}": v for k, v in json.load(open(PROBLEMS / "exam_signatures_level2.json")).items()})
    val = {r["problem_id"] for r in read_jsonl(DATASET / "final" / "val.jsonl")}
    out.update({f"val:{r['problem_id']}": r["signature"] for r in read_jsonl(PROBLEMS / "problems.jsonl") if r["problem_id"] in val and r.get("signature")})
    return out


def reason(sig: dict, ban: dict) -> str | None:
    for k, s in ban.items():
        if "op_multiset" not in s:
            continue
        j = jaccard(sig["op_multiset"], s["op_multiset"])
        if j >= BAR:  # level 1 included: filter_contamination only flags those for review; generated batches drop them outright
            return f"contamination:extra:{k}:j={j:.2f}"
        if k.startswith("exam:") and s.get("input_shapes") == sig.get("input_shapes") and j >= SHAPE_BAR:
            return f"contamination:shape:{k}:j={j:.2f}"
    return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rows, ban, dropped = read_jsonl(PROBLEMS / "problems.jsonl"), banned(), []
    for r in rows:
        if r.get("batch") != a.batch or r["status"] != "ok" or not r.get("signature") or "op_multiset" not in r["signature"]:
            continue
        why = reason(r["signature"], ban)
        if why:
            r["status"], r["exclude_reason"] = "excluded", why
            dropped.append({"problem_id": r["problem_id"], "reason": why, "at": now()})
            print("EXCLUDE", r["problem_id"], why)
    print(f"{len(dropped)} excluded in batch {a.batch}")
    if not a.dry_run and dropped:
        write_jsonl(PROBLEMS / "problems.jsonl", rows)
        write_jsonl(PROBLEMS / "excluded.jsonl", dropped, append=True)


if __name__ == "__main__":
    main()
