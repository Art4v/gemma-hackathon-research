"""Dataset v4: turn the TPU-judged batches on the team's branches into dataset-v1-format rows, one file per part under dataset/v4/.

  python dataset/scripts/v4_parts.py [--parts m1,kbk,kbkc,prod] [--exclude-file IDS.txt]

Each part is read from its own branch (git show, nothing checked out): problems.jsonl, trajectories/<batch>.jsonl, every verdicts/*.jsonl and
candidates/<batch>/*.jsonl. Rows come from assemble.examples_for, dataset-v1's own emission rule: a chain that reached a TPU `pass` gives a single row,
and a repair row if it took more than one attempt; correct_slow is never a target. Only problems whose status is `ok` on the part's branch count.
Every row goes to train (val stays dataset-v1's 26). A row from a dataset-v1 val family is an error. Writes dataset/v4/<part>.jsonl and parts_report.json.
"""
from __future__ import annotations

import argparse, collections, json, pathlib, subprocess, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from assemble import examples_for  # noqa: E402
from common import DATASET, FINAL, REPO, now, read_jsonl, write_json, write_jsonl  # noqa: E402

V4 = DATASET / "v4"
# part -> (branch, batch). Karthik's agent collects every judged round into these canonical branches.
PARTS = {"m1": ("karthik/m1", "m1"), "kbk": ("karthik/kbk", "kbk"), "kbkc": ("karthik/kbkc", "kbkc"), "prod": ("karthik/kbk", "prod"), "kbkd": ("karthik/kbk", "kbkd")}


def git_lines(ref: str, path: str) -> list:
    r = subprocess.run(["git", "-C", str(REPO), "show", f"{ref}:{path}"], capture_output=True, text=True)
    return [json.loads(l) for l in r.stdout.splitlines() if l.strip()] if r.returncode == 0 else []


def git_ls(ref: str, prefix: str) -> list:
    r = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", ref, "--", prefix], capture_output=True, text=True)
    return [p for p in r.stdout.split() if p.endswith(".jsonl") and "raw_responses" not in p]


def part_rows(name: str, branch: str, batch: str, val_families: set, excluded: set) -> tuple:
    ref = f"origin/{branch}"
    sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", ref], capture_output=True, text=True).stdout.strip()
    problems = {p["problem_id"]: p for p in git_lines(ref, "dataset/problems/problems.jsonl")}
    trajectories = git_lines(ref, f"dataset/trajectories/{batch}.jsonl")
    verdicts = {v["candidate_id"]: v for p in git_ls(ref, "dataset/verdicts") for v in git_lines(ref, p)}
    candidates = {c["candidate_id"]: c for p in git_ls(ref, f"dataset/candidates/{batch}") for c in git_lines(ref, p)}
    rows, skip, errors = [], collections.Counter(), []
    for t in trajectories:
        p = problems.get(t["problem_id"])
        if p is None or p.get("status") != "ok":
            skip["problem_not_ok"] += 1
            continue
        if t["problem_id"] in excluded:
            skip["excluded_by_list"] += 1
            continue
        if p["family_id"] in val_families:
            errors.append(f"{t['problem_id']}: in a dataset-v1 val family")
            continue
        try:
            ex = examples_for(t, p, verdicts, candidates, emit_suffix=False, human_targets=False, val_frac=0.0)
        except (KeyError, ValueError) as e:
            errors.append(f"{t['problem_id']}: {type(e).__name__} {e}")
            continue
        for e in ex:
            e["split"] = "train"
            e["meta"]["v3"] = {"origin": f"v4_{name}", "verification": "tpu", "branch": branch, "commit": sha, "batch": batch}
            rows.append(e)
    status = collections.Counter(t.get("status") for t in trajectories)
    return rows, {"branch": branch, "commit": sha, "batch": batch, "trajectories": dict(status), "rows": len(rows),
                  "by_kind": dict(collections.Counter(r["kind"] for r in rows)), "skipped": dict(skip), "errors": errors[:20], "n_errors": len(errors)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parts", default=",".join(PARTS)); ap.add_argument("--exclude-file", default="")
    a = ap.parse_args(argv)
    val_families = {e["family_id"] for e in read_jsonl(FINAL / "val.jsonl")}
    excluded = set(pathlib.Path(a.exclude_file).read_text().split()) if a.exclude_file else set()
    report = {"at": now(), "excluded_list": len(excluded)}
    for name in a.parts.split(","):
        branch, batch = PARTS[name]
        rows, rep = part_rows(name, branch, batch, val_families, excluded)
        rows.sort(key=lambda r: r["example_id"])
        write_jsonl(V4 / f"{name}.jsonl", rows)
        report[name] = rep
        print(name, json.dumps({k: rep[k] for k in ("commit", "trajectories", "rows", "by_kind", "skipped", "n_errors")}), flush=True)
    write_json(V4 / "parts_report.json", report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
