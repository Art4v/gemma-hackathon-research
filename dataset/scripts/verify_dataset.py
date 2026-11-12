"""Stage 8: check a finished final/ directory. Exit 1 on any failure, printing each one.

  python dataset/scripts/verify_dataset.py FINAL_DIR [--root dataset/]

PLAN.md "Verification", as far as it is checkable offline:
  1 schemas          every example, and every verdict and candidate record under <root>, passes common.validate
  2 verdict backing  every kernel turn has a verdict record (same candidate_id, same task) and a candidate record whose code matches
                     byte for byte; the final turn's verdict is pass; each earlier turn's is not pass and the feedback turn that follows
                     it equals that verdict's feedback byte for byte, with a matching verdict_ref
  3 exam leakage     no ref_code or code has the sha256 of any of the 73 benchmarks/jaxbench baseline files; no problem's KernelBench
                     stem is one the exam uses (the op-chain and near-dup scans are filter_contamination.py / filter_neardup.py's;
                     run them on FINAL_DIR as well)
  4 provenance       distinct (oracle_commit, oracle_sha) printed and warned about if more than one; every backing row was judged at
                     speed_bar 0.7; reverify_items.jsonl lists exactly the distinct final kernels
  5 split hygiene    no family_id in both splits; example_ids unique
  6 stats            stats.json counts and parity share match a recount
  7 seeds            every final kernel's row says seeds_distinct (its create_inputs varies across seeds)
"""
from __future__ import annotations

import argparse, collections, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402
from common import PARITY, SPEED_BAR, read_json, read_jsonl, sha256  # noqa: E402


def verify(final_dir: pathlib.Path, root: pathlib.Path) -> tuple:
    """(failures, warnings, notes)."""
    fail, warn, note = [], [], []
    train, val = read_jsonl(final_dir / "train.jsonl"), read_jsonl(final_dir / "val.jsonl")
    examples = train + val
    if not examples:
        return [f"{final_dir}: no examples in train.jsonl or val.jsonl"], warn, note
    verdicts = {v["candidate_id"]: v for p in sorted((root / "verdicts").glob("*.jsonl")) for v in read_jsonl(p)}
    candidates = {c["candidate_id"]: c for p in sorted((root / "candidates").glob("**/*.jsonl")) for c in read_jsonl(p)}
    problems = {p["problem_id"]: p for p in read_jsonl(root / "problems" / "problems.jsonl")}

    # 1 schemas
    for e in examples:
        fail += [f"schema example {e.get('example_id')}: {x}" for x in common.validate("example", e)]
    for v in verdicts.values():
        fail += [f"schema verdict {v.get('candidate_id')}: {x}" for x in common.validate("verdict", v)]
    for c in candidates.values():
        fail += [f"schema candidate {c.get('candidate_id')}: {x}" for x in common.validate("candidate", c)]

    # 2 verdict backing
    final_kernels, rows_used = {}, []
    for e in examples:
        eid, turns = e["example_id"], e["turns"]
        kernels = [(i, t) for i, t in enumerate(turns) if t.get("type") == "kernel"]
        for n, (i, t) in enumerate(kernels):
            cid, last = t.get("candidate_id"), n == len(kernels) - 1
            v, c = verdicts.get(cid), candidates.get(cid)
            if v is None:
                fail.append(f"{eid}: kernel {cid} has no verdict record")
                continue
            rows_used.append(v)
            if v["row"].get("task") != e["task_name"]:
                fail.append(f"{eid}: verdict row task {v['row'].get('task')!r} != example task {e['task_name']!r}")
            if c is None:
                fail.append(f"{eid}: kernel {cid} has no candidate record")
            elif c["code"] != t["code"] or c["code_sha256"] != sha256(t["code"]):
                fail.append(f"{eid}: kernel {cid} code differs from its candidate record")
            if last:
                if v["verdict"] != "pass":
                    fail.append(f"{eid}: final kernel {cid} verdict is {v['verdict']}, not pass")
                final_kernels[cid] = e["task_name"]
            else:
                if v["verdict"] == "pass":
                    fail.append(f"{eid}: non-final kernel {cid} passed")
                fb = turns[i + 1] if i + 1 < len(turns) else {}
                if fb.get("type") != "oracle_feedback":
                    fail.append(f"{eid}: kernel {cid} is not followed by an oracle_feedback turn")
                else:
                    if fb.get("feedback") != v["feedback"]:
                        fail.append(f"{eid}: feedback after {cid} is not the oracle's feedback byte for byte")
                    if fb.get("verdict") != v["verdict"]:
                        fail.append(f"{eid}: feedback turn verdict {fb.get('verdict')} != {v['verdict']}")
                    ref = fb.get("verdict_ref") or {}
                    if ref.get("job_id") != v["job_id"] or ref.get("oracle_sha") != v["oracle_sha"]:
                        fail.append(f"{eid}: verdict_ref after {cid} does not match the verdict record")
        if e["final_verdict"] != "pass":
            fail.append(f"{eid}: final_verdict {e['final_verdict']}")
        p = problems.get(e["problem_id"])
        if p is None:
            fail.append(f"{eid}: no problem record for {e['problem_id']}")
        elif p["ref_code"] != turns[0].get("ref_code"):
            fail.append(f"{eid}: problem turn ref_code differs from the problem record")

    # 3 exam leakage (hash and KernelBench stem layers)
    exam = {}
    for task, level, path in common.exam_files():
        exam[sha256(path.read_bytes())] = task
        exam[sha256(path.read_text())] = task
    stems = common.exam_kb_stems()
    for e in examples:
        for t in e["turns"]:
            for k in ("ref_code", "code"):
                if k in t and sha256(t[k]) in exam:
                    fail.append(f"{e['example_id']}: {k} is byte-identical to exam file {exam[sha256(t[k])]}")
        p = problems.get(e["problem_id"]) or {}
        stem = (p.get("source_ref") or {}).get("kb_stem")
        if stem and stem in stems:
            fail.append(f"{e['example_id']}: KernelBench stem {stem} is exam task {stems[stem]}")
    note.append("op-chain and near-duplicate layers are not re-run here: run filter_contamination.py and filter_neardup.py on this final/ too")

    # 4 provenance
    prov = sorted({(v["oracle_commit"], v["oracle_sha"]) for v in rows_used})
    note.append(f"oracle provenance: {prov}")
    if len(prov) > 1:
        warn.append(f"more than one (oracle_commit, oracle_sha) backs this set: {prov}")
    for v in rows_used:
        if v["row"].get("speed_bar") != SPEED_BAR:
            fail.append(f"{v['candidate_id']}: judged at speed_bar {v['row'].get('speed_bar')}, not {SPEED_BAR}")
    reverify = read_jsonl(final_dir / "reverify_items.jsonl")
    rids = {r["id"] for r in reverify}
    if rids != set(final_kernels):
        fail.append(f"reverify_items.jsonl lists {len(rids)} kernels, final kernels are {len(final_kernels)}: "
                    f"missing {sorted(set(final_kernels) - rids)[:5]}, extra {sorted(rids - set(final_kernels))[:5]}")
    for r in reverify:
        c = candidates.get(r["id"])
        if c and c["code"] != r["code"]:
            fail.append(f"reverify item {r['id']}: code differs from the candidate record")
        if final_kernels.get(r["id"]) not in (None, r["task"]):
            fail.append(f"reverify item {r['id']}: task {r['task']} != {final_kernels[r['id']]}")

    # 5 split hygiene
    fam = collections.defaultdict(set)
    for e in examples:
        fam[e["family_id"]].add(e["split"])
    both = sorted(f for f, s in fam.items() if len(s) > 1)
    if both:
        fail.append(f"family_id in both splits: {both[:10]}")
    ids = collections.Counter(e["example_id"] for e in examples)
    fail += [f"duplicate example_id {i}" for i, k in ids.items() if k > 1]
    for e in train:
        if e["split"] != "train":
            fail.append(f"{e['example_id']} in train.jsonl has split {e['split']}")
    for e in val:
        if e["split"] != "val":
            fail.append(f"{e['example_id']} in val.jsonl has split {e['split']}")

    # 6 stats
    stats = read_json(final_dir / "stats.json") if (final_dir / "stats.json").exists() else None
    if stats is None:
        fail.append("stats.json missing")
    else:
        c = collections.Counter
        want = {"n_examples": len(examples), "by_kind": dict(c(e["kind"] for e in examples)),
                "by_source": dict(c(e["source"] for e in examples)), "by_split": dict(c(e["split"] for e in examples)),
                "by_target_origin": dict(c(e["meta"]["target_origin"] for e in examples))}
        for k, v in want.items():
            if stats.get(k) != v:
                fail.append(f"stats.json {k} = {stats.get(k)} but recount gives {v}")
        finals = [e for e in examples if (e["kind"] == "single" and e["derived_from"] is None) or
                  (e["kind"] == "repair" and not e["example_id"].endswith(".suffix"))]
        share = round(sum(1 for e in finals if e["at_parity"]) / len(finals), 4) if finals else None
        if stats.get("parity_share") != share:
            fail.append(f"stats.json parity_share = {stats.get('parity_share')} but recount gives {share}")
        for e in examples:
            if e["at_parity"] != (e["speed_vs_xla"] is not None and e["speed_vs_xla"] >= PARITY):
                fail.append(f"{e['example_id']}: at_parity inconsistent with speed_vs_xla")

    # 7 seeds
    for cid in final_kernels:
        v = verdicts.get(cid)
        if v is not None and v["row"].get("seeds_distinct") is not True:
            fail.append(f"{cid}: final kernel's row has seeds_distinct={v['row'].get('seeds_distinct')}; its create_inputs does not vary across seeds")

    note.append(f"{len(examples)} examples ({len(train)} train, {len(val)} val), {len(final_kernels)} distinct final kernels")
    return fail, warn, note


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("final_dir")
    ap.add_argument("--root", default=str(common.DATASET))
    a = ap.parse_args(argv)
    fail, warn, note = verify(pathlib.Path(a.final_dir).resolve(), pathlib.Path(a.root).resolve())
    for n in note:
        print(f"note: {n}")
    for w in warn:
        print(f"warning: {w}")
    for f in fail:
        print(f"FAIL: {f}")
    print("verify_dataset: " + ("OK" if not fail else f"{len(fail)} failure(s)"))
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
