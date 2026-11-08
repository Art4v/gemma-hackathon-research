"""Stage 7: turn solved trajectories into training examples, split them, and write final/ with stats, manifest and reverify items.

  python dataset/scripts/assemble.py --phase pilot|full [--root dataset/] [--emit-suffix] [--no-human-targets] [--val-frac 0.1]

Reads  <root>/trajectories/*.jsonl, <root>/verdicts/*.jsonl, <root>/candidates/**/*.jsonl, <root>/problems/problems.jsonl
Writes <root>/final/[pilot/]{train,val,unsolved,reverify_items}.jsonl, stats.json, MANIFEST.json

Emission per problem (PROCESS.md section 7): a1 pass -> single; a2 pass -> repair [P,a1,fb1,a2] and single [P,a2]; a3 pass -> repair
[P,a1,fb1,a2,fb2,a3] and single [P,a3] (the suffix [P,a2,fb2,a3] only with --emit-suffix); a passing human kernel -> single with
target_origin human. correct_slow is never a final turn. Feedback turns are the oracle's feedback strings, verbatim.
"""
from __future__ import annotations

import argparse, collections, pathlib, statistics, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402
from common import PARITY, PIPELINE_VERSION, now, read_jsonl, sha256, write_json, write_jsonl  # noqa: E402


def load_all(root: pathlib.Path) -> tuple:
    trajectories = [t for p in sorted((root / "trajectories").glob("*.jsonl")) for t in read_jsonl(p)]
    verdicts = {v["candidate_id"]: v for p in sorted((root / "verdicts").glob("*.jsonl")) for v in read_jsonl(p)}
    candidates = {c["candidate_id"]: c for p in sorted((root / "candidates").glob("**/*.jsonl")) for c in read_jsonl(p)}
    problems = {p["problem_id"]: p for p in read_jsonl(root / "problems" / "problems.jsonl")}
    return trajectories, verdicts, candidates, problems


def split_of(family_id: str, val_frac: float) -> str:
    return "val" if int(sha256(family_id)[:8], 16) / 2 ** 32 < val_frac else "train"


def _problem_turn(problem: dict) -> dict:
    return {"role": "user", "type": "problem", "ref_code": problem["ref_code"]}


def _kernel_turn(cand: dict) -> dict:
    return {"role": "assistant", "type": "kernel", "code": cand["code"], "candidate_id": cand["candidate_id"]}


def _feedback_turn(v: dict) -> dict:
    return {"role": "user", "type": "oracle_feedback", "feedback": v["feedback"], "verdict": v["verdict"],
            "verdict_ref": {"job_id": v["job_id"], "oracle_sha": v["oracle_sha"]}}


def _example(problem: dict, chain: list, verdicts: dict, candidates: dict, kind: str, example_id: str, derived_from: str | None,
             target_origin: str, split: str) -> dict:
    """`chain` is the ordered list of candidate_ids whose last entry passed."""
    turns = [_problem_turn(problem)]
    for i, cid in enumerate(chain):
        turns.append(_kernel_turn(candidates[cid]))
        if i < len(chain) - 1:
            turns.append(_feedback_turn(verdicts[cid]))
    final, vs = candidates[chain[-1]], [verdicts[c] for c in chain]
    speed = vs[-1]["speed_vs_xla"]
    c = final.get("constructor") or {}
    return {"example_id": example_id, "kind": kind, "derived_from": derived_from, "problem_id": problem["problem_id"],
            "family_id": problem["family_id"], "source": problem["source"], "task_name": problem["task_name"], "turns": turns,
            "final_verdict": vs[-1]["verdict"], "speed_vs_xla": speed, "at_parity": speed is not None and speed >= PARITY,
            "n_repairs": len(chain) - 1,
            "oracle": {"commits": sorted({v["oracle_commit"] for v in vs}), "shas": sorted({v["oracle_sha"] for v in vs}),
                       "job_ids": sorted({v["job_id"] for v in vs})},
            "meta": {"plan": final.get("plan", ""), "target_origin": target_origin,
                     "constructor": {k: c.get(k) for k in ("backend", "model", "prompt_version")},
                     "failing_verdicts": [v["verdict"] for v in vs[:-1]]},
            "split": split}


def examples_for(traj: dict, problem: dict, verdicts: dict, candidates: dict, *, emit_suffix: bool, human_targets: bool,
                 val_frac: float) -> list:
    out, task, split = [], traj["task_name"], split_of(problem["family_id"], val_frac)
    attempts = sorted(traj.get("attempts", []), key=lambda a: (a["attempt"], a["sample"]))
    by_attempt = {}
    for a in attempts:  # one candidate per attempt number: the last sample is the one the chain used
        by_attempt[a["attempt"]] = a
    chain_ids = [by_attempt[k]["candidate_id"] for k in sorted(by_attempt)]
    passing = [k for k in sorted(by_attempt) if by_attempt[k]["verdict"] == "pass"]
    if traj.get("status") == "solved" and passing:
        k = passing[0]
        chain = chain_ids[: chain_ids.index(by_attempt[k]["candidate_id"]) + 1]
        if any(verdicts[c]["verdict"] == "pass" for c in chain[:-1]):
            raise ValueError(f"{task}: a non-final attempt passed; the chain is inconsistent")
        if k == 1:
            out.append(_example(problem, chain, verdicts, candidates, "single", f"{task}.single.a1", None, "constructor", split))
        else:
            rid = f"{task}.repair.a{k}"
            out.append(_example(problem, chain, verdicts, candidates, "repair", rid, None, "constructor", split))
            out.append(_example(problem, chain[-1:], verdicts, candidates, "single", f"{task}.single.a{k}", rid, "constructor", split))
            if k == 3 and emit_suffix:
                out.append(_example(problem, chain[1:], verdicts, candidates, "repair", f"{rid}.suffix", rid, "constructor", split))
    human = traj.get("human")
    if human_targets and human and human["verdict"] == "pass":
        out.append(_example(problem, [human["candidate_id"]], verdicts, candidates, "single", f"{task}.single.human", None, "human", split))
    return out


def _quantiles(xs: list) -> dict | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return {"p10": xs[0], "p50": xs[0], "p90": xs[0], "n": 1}
    q = statistics.quantiles(xs, n=10, method="inclusive")
    return {"p10": q[0], "p50": statistics.median(xs), "p90": q[-1], "n": len(xs)}


def stats_for(examples: list, trajectories: list, problems: dict) -> dict:
    c = collections.Counter
    finals = [e for e in examples if e["kind"] == "single" and e["derived_from"] is None or e["kind"] == "repair" and not e["example_id"].endswith(".suffix")]
    speeds = [e["speed_vs_xla"] for e in finals]
    repaired = collections.defaultdict(c)
    for e in examples:
        if e["kind"] == "repair" and not e["example_id"].endswith(".suffix"):
            for i, v in enumerate(e["meta"]["failing_verdicts"], start=1):
                repaired[f"attempt{i}"][v] += 1
    # pass@k per source: share of problems (with at least one oracle-judged attempt) solved by attempt <= k
    per_source = collections.defaultdict(list)
    for t in trajectories:
        p = problems.get(t["problem_id"])
        if p is None or not t.get("attempts"):
            continue
        solved_at = min((a["attempt"] for a in t["attempts"] if a["verdict"] == "pass"), default=None)
        per_source[p["source"]].append(solved_at)
    pass_at = {src: {f"pass@{k}": round(sum(1 for s in xs if s is not None and s <= k) / len(xs), 4) for k in (1, 2, 3)}
               for src, xs in sorted(per_source.items())}
    pass_at["all"] = {f"pass@{k}": round(sum(1 for xs in per_source.values() for s in xs if s is not None and s <= k) /
                                         max(1, sum(len(xs) for xs in per_source.values())), 4) for k in (1, 2, 3)}
    return {"n_examples": len(examples), "by_kind": dict(c(e["kind"] for e in examples)),
            "by_source": dict(c(e["source"] for e in examples)), "by_split": dict(c(e["split"] for e in examples)),
            "by_n_repairs": {str(k): v for k, v in sorted(c(e["n_repairs"] for e in examples).items())},
            "by_target_origin": dict(c(e["meta"]["target_origin"] for e in examples)),
            "n_chains": len(finals), "repaired_verdicts": {k: dict(v) for k, v in sorted(repaired.items())},
            "speed_vs_xla": _quantiles(speeds),
            "parity_share": round(sum(1 for e in finals if e["at_parity"]) / len(finals), 4) if finals else None,
            "code_chars": _quantiles([len(e["turns"][-1]["code"]) for e in examples]),
            "pass_at_k": pass_at, "n_problems_judged": sum(len(v) for v in per_source.values()),
            "n_unsolved": sum(1 for t in trajectories if t.get("status") == "unsolved"),
            "n_harness_blocked": sum(1 for t in trajectories if t.get("status") == "harness_blocked")}


def assemble(root: pathlib.Path, phase: str, *, emit_suffix: bool, human_targets: bool, val_frac: float) -> int:
    trajectories, verdicts, candidates, problems = load_all(root)
    out_dir = root / "final" / ("pilot" if phase == "pilot" else "")
    examples, unsolved, errors = [], [], []
    for t in trajectories:
        p = problems.get(t["problem_id"])
        if p is None:
            errors.append(f"{t['problem_id']}: no problem record")
            continue
        try:
            examples += examples_for(t, p, verdicts, candidates, emit_suffix=emit_suffix, human_targets=human_targets, val_frac=val_frac)
        except (KeyError, ValueError) as e:
            errors.append(f"{t['problem_id']}: {e}")
        if t.get("status") in ("unsolved", "harness_blocked"):
            unsolved.append({"problem_id": t["problem_id"], "task_name": t["task_name"], "status": t["status"], "source": p["source"],
                             "attempts": [{k: a[k] for k in ("attempt", "candidate_id", "verdict", "feedback", "speed_vs_xla")} for a in t["attempts"]],
                             "harness_events": [{k: a[k] for k in ("attempt", "candidate_id", "verdict", "feedback")} for a in t.get("harness_events", [])]})
    for e in examples:
        errors += [f"{e['example_id']}: {x}" for x in common.validate("example", e)]
    ids = collections.Counter(e["example_id"] for e in examples)
    errors += [f"duplicate example_id {i}" for i, k in ids.items() if k > 1]
    if errors:
        print(f"assemble: {len(errors)} error(s); nothing written", file=sys.stderr)
        for x in errors[:50]:
            print(f"  - {x}", file=sys.stderr)
        return 1
    examples.sort(key=lambda e: e["example_id"])
    train, val = [e for e in examples if e["split"] == "train"], [e for e in examples if e["split"] == "val"]
    seen, reverify = set(), []
    for e in examples:
        k = e["turns"][-1]
        if k["candidate_id"] not in seen:
            seen.add(k["candidate_id"])
            reverify.append({"id": k["candidate_id"], "task": e["task_name"], "ref_code": e["turns"][0]["ref_code"], "code": k["code"]})
    stats = stats_for(examples, trajectories, problems)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    write_jsonl(out_dir / "unsolved.jsonl", sorted(unsolved, key=lambda u: u["problem_id"]))
    write_jsonl(out_dir / "reverify_items.jsonl", reverify)
    write_json(out_dir / "stats.json", stats)
    files = ["train.jsonl", "val.jsonl", "unsolved.jsonl", "reverify_items.jsonl", "stats.json"]
    manifest = {"phase": phase, "pipeline_version": PIPELINE_VERSION, "created_at": now(),
                "oracle_commits": sorted({c for e in examples for c in e["oracle"]["commits"]}),
                "oracle_shas": sorted({s for e in examples for s in e["oracle"]["shas"]}),
                "prompt_versions": sorted({str(e["meta"]["constructor"].get("prompt_version")) for e in examples}),
                "speed_bar": common.SPEED_BAR, "parity": PARITY, "val_frac": val_frac, "emit_suffix": emit_suffix,
                "human_targets": human_targets, "n_train": len(train), "n_val": len(val), "n_reverify": len(reverify),
                "files": {f: sha256((out_dir / f).read_bytes()) for f in files}}
    write_json(out_dir / "MANIFEST.json", manifest)
    print(f"{out_dir}: {len(train)} train, {len(val)} val, {len(unsolved)} unsolved, {len(reverify)} kernels to reverify; "
          f"parity share {stats['parity_share']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--phase", required=True, choices=("pilot", "full"))
    ap.add_argument("--root", default=str(common.DATASET))
    ap.add_argument("--emit-suffix", action="store_true", help="also emit [P, a2, fb2, a3] for chains solved at attempt 3")
    ap.add_argument("--no-human-targets", action="store_true", help="do not emit passing human kernels as examples")
    ap.add_argument("--val-frac", type=float, default=0.1)
    a = ap.parse_args(argv)
    return assemble(pathlib.Path(a.root).resolve(), a.phase, emit_suffix=a.emit_suffix, human_targets=not a.no_human_targets,
                    val_frac=a.val_frac)


if __name__ == "__main__":
    sys.exit(main())
