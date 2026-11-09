"""Dataset v3: run dataset-v1's two contamination filters, unmodified and at the same thresholds, over every new reference, then write the rows.

  python dataset/scripts/v3_finalize.py [--target 1500] [--python PY]

Inputs: generated_candidates.jsonl + generated_verified.jsonl, shape_variants_unfiltered.jsonl. filter_contamination.run: KernelBench-name and op-chain match
against the exam; an op-multiset Jaccard >= 0.8 is only flagged for human review there, and is dropped here, as dataset-v1 did with its three flagged problems.
filter_neardup.run: near-duplicate text against all 73 exam files, and duplicates within the set. Writes generated.jsonl, shape_variants.jsonl and
contamination_report.json, which records the highest similarity to JAXBench among the rows kept.
"""
from __future__ import annotations

import argparse, ast, collections, multiprocessing, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import filter_contamination as fc  # noqa: E402
import filter_neardup as fn  # noqa: E402
from common import FINAL, PROBLEMS, exam_files, exam_kb_stems, now, read_json, read_jsonl, sha256, write_json, write_jsonl  # noqa: E402
from signature import exam_signatures, signature_of  # noqa: E402
from v3_common import ORACLE_SHA, V3  # noqa: E402


_CORPUS: list = []
_REAL_NEAREST = fn.nearest


def _init_worker(corpus: list) -> None:
    _CORPUS[:] = corpus


def _nearest(tokens: list) -> list:
    return _REAL_NEAREST(tokens, _CORPUS)


def level2_signatures(python: str | None) -> dict:
    """{"l2:<task>": signature} for JAXBench's level 2 references. dataset-v1's op-chain layer read level 1 only (signature.exam_signatures); issue 5 asked for both."""
    path = V3 / "exam_signatures_level2.json"
    if path.exists():
        return read_json(path)
    out = {f"l2:{task}": signature_of(p.read_text(), python) for task, level, p in exam_files() if level == "level2"}
    write_json(path, out)
    return out


def val_problems(python: str | None) -> tuple:
    """({val task: signature}, {val task: reference}) for dataset-v1's val problems."""
    problems = {p["problem_id"]: p for p in read_jsonl(PROBLEMS / "problems.jsonl")}
    sigs, refs = {}, {}
    for e in read_jsonl(FINAL / "val.jsonl"):
        p = problems[e["problem_id"]]
        refs[e["task_name"]] = p["ref_code"]
        sig = p.get("signature")
        sigs[e["task_name"]] = sig if sig and "op_chain" in sig else signature_of(p["ref_code"], python)
    return sigs, refs


def skeleton(code: str) -> str:
    """A hash of the kernel's AST with identifiers, attribute names, literals and shapes abstracted. Keyword names stay: they are the API structure."""
    tree = ast.parse(code)
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            n.id = "N"
        elif isinstance(n, ast.Attribute):
            n.attr = "A"
        elif isinstance(n, ast.Constant):
            n.value = 0
        elif isinstance(n, ast.arg):
            n.arg = "a"
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            n.name = "f"
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            n.names = []
    return sha256(ast.dump(tree))[:12]


def record(pid: str, source: str, ref: str, signature: dict) -> dict:
    return {"problem_id": pid, "source": source, "status": "ok", "ref_code": ref, "signature": signature, "source_ref": {}, "exclude_reason": None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target", type=int, default=1500); ap.add_argument("--python", default=None)
    ap.add_argument("--partial", action="store_true", help="filter only what has been verified so far (a dry run of this step)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cap-per-skeleton", type=int, default=0, help="keep at most this many generated rows per structural skeleton (0 = no cap)")
    a = ap.parse_args(argv)
    cands = {c["id"]: c for c in read_jsonl(V3 / "generated_candidates.jsonl")}
    ver = {r["id"]: r for r in read_jsonl(V3 / "generated_verified.jsonl")}
    missing = [c for c in cands if c not in ver]
    if a.partial:
        cands = {k: v for k, v in cands.items() if k in ver}
    elif missing:
        sys.exit(f"{len(missing)} candidates are not verified yet")
    fails = collections.Counter(r.get("stage") for r in ver.values() if not r.get("ok"))
    good = [cands[i] for i, r in ver.items() if r.get("ok")]
    variants = read_jsonl(V3 / "shape_variants_unfiltered.jsonl")
    recs = [record(c["id"], "generated", c["ref_code"], ver[c["id"]]["signature"]) for c in good]
    recs += [record(v["problem_id"], "shape_variant", v["turns"][0]["ref_code"], v["signature"]) for v in variants]
    print(f"filtering {len(good)} generated and {len(variants)} shape-variant references", flush=True)
    exam_sigs = {**exam_signatures(python=a.python), **level2_signatures(a.python)}
    print(f"exam signatures: {sum('error' not in s for s in exam_sigs.values())} of {len(exam_sigs)} traced, level 1 + 2", flush=True)
    recs, ex1, review, sig1 = fc.run(recs, exam_kb_stems(), exam_sigs, a.python, log=lambda m: None)
    flagged = {r["problem_id"]: r for r in review}
    for r in recs:  # the "review" tier: dropped without asking, the same call Hruday made for dataset-v1's three
        if r["problem_id"] in flagged and r["status"] == "ok":
            r["status"], r["exclude_reason"] = "excluded", f"contamination:review:{flagged[r['problem_id']]['nearest_exam']}:j={flagged[r['problem_id']]['jaccard']}"
    print(f"layers 1-2: excluded {len(ex1)}, review-tier dropped {len(flagged)}", flush=True)
    # Issue 5 (Karthik's agent, Nov 9): the generator's families overlap val as well as JAXBench, and val is the only selection signal. So the same filters run
    # against the val problems, at the same thresholds, and a hit is dropped (val itself never changes).
    val_sigs, val_refs = val_problems(a.python)
    before = {r["problem_id"] for r in recs if r["status"] == "ok"}
    recs, _, val_review, _ = fc.run(recs, {}, val_sigs, a.python, log=lambda m: None)
    val_flagged = {r["problem_id"]: r for r in val_review}
    for r in recs:
        if r["problem_id"] in before and r["status"] != "ok":
            r["exclude_reason"] = "val_" + r["exclude_reason"]
        elif r["problem_id"] in val_flagged and r["status"] == "ok":
            r["status"], r["exclude_reason"] = "excluded", f"val_contamination:review:{val_flagged[r['problem_id']]['nearest_exam']}:j={val_flagged[r['problem_id']]['jaccard']}"
    # shape leakage: the same ops at the same input shapes as a JAXBench task would let a model memorize a tiling tuned to a test shape
    shape_hits = 0
    for r in recs:
        if r["status"] != "ok":
            continue
        for task, s in exam_sigs.items():
            if "error" not in s and s.get("op_multiset") == r["signature"].get("op_multiset") and s.get("input_shapes") == r["signature"].get("input_shapes"):
                r["status"], r["exclude_reason"] = "excluded", f"contamination:shape:{task}"
                shape_hits += 1
                break
    print(f"against val: excluded {sum(1 for r in recs if str(r.get('exclude_reason')).startswith('val_'))} (op chain or review tier); shape leakage: {shape_hits}", flush=True)
    corpus = fn.exam_corpus() + [(f"val:{t}", "val", fn.normalise(code)) for t, code in sorted(val_refs.items())]
    # filter_neardup runs as it stands. Its nearest() is pure and takes seconds per reference in Python, so its answers are computed across processes first and
    # served from this table: same function, same inputs, same results. No JAX in the workers, so each stays small.
    streams = [fn.normalise(r["ref_code"]) for r in recs if r["status"] == "ok"]
    with multiprocessing.Pool(a.workers, initializer=_init_worker, initargs=(corpus,)) as pool:
        table = dict(zip(map(tuple, streams), pool.imap(_nearest, streams, chunksize=8)))
    fn.nearest = lambda tokens, _corpus, k=3: table[tuple(tokens)][:k]
    recs, ex2, sig2, worst = fn.run(recs, corpus, V3 / "none.jsonl", log=lambda m: None)
    print(f"layer 3: excluded {len(ex2)}; max similarity seen jaccard {worst[0]:.3f} ratio {worst[1]:.3f} ({worst[2]})", flush=True)
    status = {r["problem_id"]: r for r in recs}
    reasons = collections.Counter(r["exclude_reason"].split(":")[0] + ":" + r["exclude_reason"].split(":")[1] for r in recs if r["status"] != "ok" and ":" in r["exclude_reason"])
    survivors = [c for c in good if status[c["id"]]["status"] == "ok"]
    skel = {c["id"]: skeleton(c["code"]) for c in survivors}
    if a.cap_per_skeleton:
        taken, capped = collections.Counter(), []
        for c in sorted(survivors, key=lambda c: sha256(c["id"])):
            taken[skel[c["id"]]] += 1
            if taken[skel[c["id"]]] <= a.cap_per_skeleton:
                capped.append(c)
        survivors = capped
    by_fam = collections.defaultdict(list)
    for c in sorted(survivors, key=lambda c: sha256(c["id"])):
        by_fam[c["family"]].append(c)
    # the same mix as the survivors, cut to the target by a stable hash order
    frac = min(1.0, a.target / max(1, len(survivors)))
    keep = [c for fam, xs in sorted(by_fam.items()) for c in xs[: round(len(xs) * frac)]]
    rows = []
    for c in keep:
        r = ver[c["id"]]
        nearest = sig1.get(c["id"], {}).get("nearest_exam_opchain")
        text = (sig2.get(c["id"], {}).get("nearest_exam_text") or {}).get("ref", [])
        rows.append({"example_id": f"{c['id']}.generated", "kind": "single", "derived_from": None, "problem_id": c["id"], "family_id": f"generated/{c['id']}", "source": "generated", "task_name": c["id"],
                     "turns": [{"role": "user", "type": "problem", "ref_code": c["ref_code"]}, {"role": "assistant", "type": "kernel", "code": c["code"], "candidate_id": c["id"]}],
                     "final_verdict": "cpu_verified", "speed_vs_xla": None, "at_parity": False, "n_repairs": 0, "oracle": {"commits": [], "shas": [ORACLE_SHA], "job_ids": ["v3-cpu"]}, "split": "train",
                     "meta": {"plan": "", "target_origin": "generator", "constructor": {"backend": "template", "model": None, "prompt_version": None}, "failing_verdicts": [],
                              "v3": {"origin": "generated", "verification": "cpu", "family": c["family"], "skeleton": skel[c["id"]], "target_static_verdict": r["static_verdict"], "lowered": r["lowered"], "ref_precision": r["ref_precision"],
                                     "numerics": r["per_seed"], "nearest_exam": {"opchain": nearest, "text": text[:1]}}}})
    write_jsonl(V3 / "generated.jsonl", rows)
    kept_v = []
    for v in variants:
        if status[v["problem_id"]]["status"] == "ok":
            v = dict(v); v.pop("signature", None)
            v["meta"]["v3"]["nearest_exam"] = {"opchain": sig1.get(v["problem_id"], {}).get("nearest_exam_opchain"), "text": (sig2.get(v["problem_id"], {}).get("nearest_exam_text") or {}).get("ref", [])[:1]}
            kept_v.append(v)
    write_jsonl(V3 / "shape_variants.jsonl", kept_v)
    chosen = {c["id"] for c in keep} | {v["problem_id"] for v in kept_v}
    op = [(sig1[i]["nearest_exam_opchain"]["jaccard"], i, sig1[i]["nearest_exam_opchain"]["task"]) for i in chosen if i in sig1]
    txt = [((sig2[i]["nearest_exam_text"]["ref"] or [{}])[0].get("jaccard", 0), (sig2[i]["nearest_exam_text"]["ref"] or [{}])[0].get("ratio", 0), i, (sig2[i]["nearest_exam_text"]["ref"] or [{}])[0].get("task")) for i in chosen if i in sig2]
    top_by_level = lambda level, key: max(((n[key], i, n["task"]) for i in chosen if i in sig2 for n in sig2[i]["nearest_exam_text"]["ref"] if n["level"] == level), default=None)
    kept_skel = collections.Counter(skel[c["id"]] for c in keep)
    diversity = {"skeletons_distinct": len(kept_skel), "top10_share": round(sum(v for _, v in kept_skel.most_common(10)) / max(1, len(keep)), 4),
                 "largest_skeleton_rows": max(kept_skel.values(), default=0), "cap_per_skeleton": a.cap_per_skeleton, "survivors_skeletons_distinct": len(set(skel.values()))}
    report = {"at": now(), "diversity": diversity, "max_text_vs_val": {"jaccard": top_by_level("val", "jaccard"), "ratio": top_by_level("val", "ratio")}, "thresholds": {"op_multiset_jaccard_dropped_at": fc.REVIEW_JACCARD, "text_jaccard": fn.JACCARD_BAR, "text_ratio": fn.RATIO_BAR, "within_set_duplicate_jaccard": fn.DUP_JACCARD},
              "generated_candidates": len(cands), "generated_verify_ok": len(good), "generated_verify_failed_at": dict(fails), "generated_after_filters": len(survivors), "generated_kept": len(keep),
              "generated_kept_by_family": dict(collections.Counter(c["family"] for c in keep)), "shape_variants_in": len(variants), "shape_variants_kept": len(kept_v),
              "excluded_by_reason": dict(reasons), "max_similarity_kept": {"op_multiset_jaccard": max(op, default=None), "text_jaccard": max((t[0], t[2], t[3]) for t in txt) if txt else None,
                                                                         "text_ratio": max((t[1], t[2], t[3]) for t in txt) if txt else None}}
    write_json(V3 / "contamination_report.json", report)
    print({k: report[k] for k in ("generated_verify_ok", "generated_after_filters", "generated_kept", "generated_kept_by_family", "shape_variants_kept", "excluded_by_reason", "max_similarity_kept", "max_text_vs_val", "diversity")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
