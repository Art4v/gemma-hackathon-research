"""Stage 3, layers 1 and 2 of the contamination filter (PROCESS.md section 3): KernelBench-stem match and op-chain match against the
exam. Layer 3 (near-duplicate text) is filter_neardup.py.

  python dataset/scripts/filter_contamination.py [--problems dataset/problems/problems.jsonl] [--dry-run] [--python PY]

Reads every `ok` problem, computes a missing `signature` (signature.py, in an isolated child), and excludes:
  contamination:name:<exam task>     a kernelbench record whose KernelBench stem is one the exam uses (number ignored too)
  contamination:opchain:<exam task>  op_chain equal to an exam reference's, whatever the shapes
Records with op-multiset Jaccard >= REVIEW_JACCARD against an exam task go to review.jsonl (kept, for a human). Rewrites problems.jsonl,
appends excluded.jsonl, merges signatures.jsonl (per problem: signature, nearest exam task by multiset Jaccard).
"""
from __future__ import annotations

import argparse, pathlib, re, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import HBM_BUDGET_BYTES, PROBLEMS, exam_kb_stems, now, read_jsonl, write_jsonl  # noqa: E402
from signature import exam_signatures, jaccard, signature_of  # noqa: E402

REVIEW_JACCARD = 0.8


def strip_number(stem: str) -> str:
    return re.sub(r"^\d+_", "", stem).lower()


def name_match(kb_stem: str | None, exam_stems: dict) -> str | None:
    """The exam task whose KernelBench stem equals `kb_stem` (exactly, or ignoring the leading number), else None."""
    if not kb_stem:
        return None
    if kb_stem in exam_stems:
        return exam_stems[kb_stem]
    bare = strip_number(kb_stem)
    for s, task in exam_stems.items():
        if strip_number(s) == bare:
            return task
    return None


def opchain_match(sig: dict, exam_sigs: dict) -> str | None:
    chain = sig.get("op_chain")
    if not chain:
        return None
    for task, es in exam_sigs.items():
        if "error" not in es and es.get("op_chain") == chain:
            return task
    return None


def nearest(sig: dict, exam_sigs: dict) -> tuple:
    """(exam task, Jaccard) of the closest exam op-multiset."""
    best, bj = None, 0.0
    for task, es in exam_sigs.items():
        if "error" in es:
            continue
        j = jaccard(sig.get("op_multiset", {}), es.get("op_multiset", {}))
        if j > bj:
            best, bj = task, j
    return best, bj


def merge_signatures(path: pathlib.Path, updates: dict) -> None:
    """signatures.jsonl keyed by problem_id; `updates` maps problem_id -> fields to merge."""
    rows = {r["problem_id"]: r for r in read_jsonl(path)}
    for pid, fields in updates.items():
        rows.setdefault(pid, {"problem_id": pid}).update(fields)
    write_jsonl(path, [rows[k] for k in sorted(rows)])


def run(problems: list, exam_stems: dict, exam_sigs: dict, python: str | None = None, log=print) -> tuple:
    """(problems with status updated, excluded rows, review rows, signature updates). Pure given the inputs except for signature_of."""
    excluded, review, sig_updates = [], [], {}
    for rec in problems:
        if rec.get("status") != "ok":
            continue
        pid = rec["problem_id"]
        task = name_match(rec.get("source_ref", {}).get("kb_stem"), exam_stems) if rec.get("source") == "kernelbench" else None
        if task:
            rec["status"], rec["exclude_reason"] = "excluded", f"contamination:name:{task}"
            excluded.append({"problem_id": pid, "reason": rec["exclude_reason"], "at": now()})
            log(f"EXCLUDE {pid}: {rec['exclude_reason']}")
            continue
        sig = rec.get("signature")
        if not sig or "op_chain" not in sig:
            sig = signature_of(rec["ref_code"], python)
            rec["signature"] = sig
            if "error" not in sig:
                rec["input_bytes"] = sig["input_bytes"]
                rec["fits_hbm"] = 3 * sig["input_bytes"] <= HBM_BUDGET_BYTES
        if "error" in sig:
            rec["status"], rec["exclude_reason"] = "excluded", f"ref_broken:{sig['error'][:120]}"
            excluded.append({"problem_id": pid, "reason": rec["exclude_reason"], "at": now()})
            log(f"EXCLUDE {pid}: {rec['exclude_reason']}")
            continue
        task = opchain_match(sig, exam_sigs)
        if task:
            rec["status"], rec["exclude_reason"] = "excluded", f"contamination:opchain:{task}"
            excluded.append({"problem_id": pid, "reason": rec["exclude_reason"], "at": now()})
            log(f"EXCLUDE {pid}: {rec['exclude_reason']}")
            continue
        near, j = nearest(sig, exam_sigs)
        sig_updates[pid] = {"signature": sig, "nearest_exam_opchain": {"task": near, "jaccard": round(j, 3)}}
        if j >= REVIEW_JACCARD:
            review.append({"problem_id": pid, "nearest_exam": near, "jaccard": round(j, 3), "op_chain": sig["op_chain"], "at": now()})
            log(f"REVIEW  {pid}: multiset Jaccard {j:.2f} with {near}")
    return problems, excluded, review, sig_updates


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--problems", default=str(PROBLEMS / "problems.jsonl"))
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--python", default=None)
    a = ap.parse_args(argv)
    path = pathlib.Path(a.problems)
    problems = read_jsonl(path)
    if not problems:
        sys.exit(f"no problems in {path}")
    exam_stems, exam_sigs = exam_kb_stems(), exam_signatures(python=a.python)
    problems, excluded, review, sig_updates = run(problems, exam_stems, exam_sigs, a.python)
    kept = sum(1 for p in problems if p["status"] == "ok")
    by_reason: dict = {}
    for e in excluded:
        k = e["reason"].split(":")[0] + (":" + e["reason"].split(":")[1] if e["reason"].startswith("contamination") else "")
        by_reason[k] = by_reason.get(k, 0) + 1
    print(f"kept {kept} of {len(problems)}; excluded {by_reason}; review {len(review)}; exam refs traced {sum('error' not in s for s in exam_sigs.values())}/{len(exam_sigs)}")
    if a.dry_run:
        return
    write_jsonl(path, problems)
    write_jsonl(path.parent / "excluded.jsonl", excluded, append=True)
    write_jsonl(path.parent / "review.jsonl", review)
    merge_signatures(path.parent / "signatures.jsonl", sig_updates)


if __name__ == "__main__":
    main()
