"""Export the finished dataset in two layouts: JAXBench-style folders and a JSONL index.

  python dataset/scripts/export.py [--val-frac 0.1]      writes dataset/final/
  python dataset/scripts/export.py --check               verifies dataset/final/ against itself; exit 1 on any mismatch

Included: every problem with status ok whose cpu_check passed and, for scraped problems, whose semantics_check passed.
Everything else goes to final/excluded.jsonl with its reason.

  final/problems/<task_name>/baseline.py   the reference module, laid out like benchmarks/jaxbench/level1/<task>/baseline.py
  final/problems.jsonl                     one record per problem (fields: README.md)
  final/excluded.jsonl                     every problem left out, with the reason
  final/manifest.json                      counts, sha256 of every file, jax version, oracle sha, pipeline version
"""
from __future__ import annotations

import argparse, collections, json, pathlib, shutil, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import FINAL, PIPELINE_VERSION, PROBLEMS, now, read_json, read_jsonl, sha256, write_json, write_jsonl  # noqa: E402


def split_of(family_id: str, val_frac: float) -> str:
    return "val" if int(sha256(family_id)[:8], 16) / 2 ** 32 < val_frac else "train"


def reason_excluded(r: dict) -> str | None:
    if r["status"] != "ok":
        return r.get("exclude_reason") or "excluded"
    cc = r.get("cpu_check")
    if cc is None:
        return "cpu_check: not run"
    if not cc.get("passed"):
        return f"cpu_check: failed {','.join(cc.get('failed_checks', []))} {cc.get('error', '')}".strip()
    if r["source"] == "scraped" and not (r["source_ref"].get("semantics_check") or {}).get("ok"):
        return "semantics_check: not passed"
    return None


def record(r: dict, sig: dict, val_frac: float) -> dict:
    s, cc = r.get("signature") or {}, r["cpu_check"]
    near = ((sig.get("nearest_exam_text") or {}).get("ref") or [{}])[0]
    src = {k: v for k, v in r["source_ref"].items() if k != "semantics_check"}
    return {"task_name": r["task_name"], "problem_id": r["problem_id"], "family_id": r["family_id"], "source": r["source"], "source_ref": src,
            "path": f"problems/{r['task_name']}/baseline.py", "ref_sha256": r["ref_sha256"],
            "input_shapes": cc.get("shapes") or s.get("input_shapes"), "input_dtypes": s.get("input_dtypes"),
            "output_shapes": cc.get("output_shapes") or s.get("output_shapes"), "output_dtypes": cc.get("output_dtypes"),
            "input_bytes": cc.get("input_bytes", r.get("input_bytes")), "op_chain": s.get("op_chain"),
            "description": r.get("plan", ""),
            "cpu_check": {k: cc[k] for k in ("passed", "loads", "fits_hbm", "deterministic", "seeds_distinct", "runs", "finite", "highest_precision",
                                              "run_s", "peak_rss_gb", "err_default_vs_highest", "jax", "oracle_sha", "checked_at") if k in cc},
            "port_check": r.get("port_check"), "semantics_check": r["source_ref"].get("semantics_check"),
            "nearest_exam": {"text": {k: near.get(k) for k in ("task", "jaccard", "ratio")} if near else None,
                             "op_chain": sig.get("nearest_exam_opchain")},
            "split": split_of(r["family_id"], val_frac)}


def export(val_frac: float) -> dict:
    rows = read_jsonl(PROBLEMS / "problems.jsonl")
    sigs = {s["problem_id"]: s for s in read_jsonl(PROBLEMS / "signatures.jsonl")}
    if FINAL.exists():
        shutil.rmtree(FINAL)
    (FINAL / "problems").mkdir(parents=True)
    keep, drop = [], []
    for r in sorted(rows, key=lambda r: r["problem_id"]):
        why = reason_excluded(r)
        if why:
            drop.append({"problem_id": r["problem_id"], "source": r["source"], "reason": why}); continue
        d = FINAL / "problems" / r["task_name"]
        d.mkdir(parents=True)
        (d / "baseline.py").write_text(r["ref_code"])
        keep.append(record(r, sigs.get(r["problem_id"], {}), val_frac))
    for e in read_jsonl(PROBLEMS / "excluded.jsonl"):  # dropped before reaching problems.jsonl (port, author, wrap failures)
        if e.get("problem_id") not in {d["problem_id"] for d in drop} | {k["problem_id"] for k in keep}:
            drop.append({"problem_id": e["problem_id"], "source": e.get("source"), "reason": e.get("exclude_reason") or e.get("reason")})
    write_jsonl(FINAL / "problems.jsonl", keep)
    write_jsonl(FINAL / "excluded.jsonl", drop)
    files = {str(p.relative_to(FINAL)): sha256(p.read_bytes()) for p in sorted(FINAL.rglob("*")) if p.is_file()}
    cc = [k["cpu_check"] for k in keep]
    m = {"n_problems": len(keep), "n_excluded": len(drop), "by_source": dict(collections.Counter(k["source"] for k in keep)),
         "by_split": dict(collections.Counter(k["split"] for k in keep)), "val_frac": val_frac,
         "jax": sorted({c.get("jax") for c in cc if c.get("jax")}), "oracle_sha": sorted({c.get("oracle_sha") for c in cc if c.get("oracle_sha")}),
         "pipeline_version": PIPELINE_VERSION, "created_at": now(), "files": files}
    write_json(FINAL / "manifest.json", m)
    return m


def check() -> list:
    bad = []
    recs = read_jsonl(FINAL / "problems.jsonl")
    m = read_json(FINAL / "manifest.json")
    dirs = {p.name for p in (FINAL / "problems").iterdir() if p.is_dir()}
    if dirs != {r["task_name"] for r in recs}:
        bad.append(f"folders {len(dirs)} != records {len(recs)}")
    for r in recs:
        p = FINAL / r["path"]
        if not p.exists():
            bad.append(f"{r['task_name']}: missing {r['path']}"); continue
        if sha256(p.read_text()) != r["ref_sha256"]:
            bad.append(f"{r['task_name']}: baseline.py sha does not match ref_sha256")
        if not r["cpu_check"].get("passed"):
            bad.append(f"{r['task_name']}: cpu_check not passed")
        if r["source"] == "scraped" and not (r.get("semantics_check") or {}).get("ok"):
            bad.append(f"{r['task_name']}: semantics_check not passed")
    if len({r["task_name"] for r in recs}) != len(recs):
        bad.append("duplicate task_name")
    fam = collections.defaultdict(set)
    for r in recs:
        fam[r["family_id"]].add(r["split"])
    bad += [f"family {f} in both splits" for f, s in fam.items() if len(s) > 1]
    for rel, h in m["files"].items():
        if rel != "manifest.json" and (not (FINAL / rel).exists() or sha256((FINAL / rel).read_bytes()) != h):
            bad.append(f"manifest sha mismatch: {rel}")
    if m["n_problems"] != len(recs):
        bad.append("manifest n_problems != records")
    if any((FINAL / "problems").rglob("*.kernel.py")) or any("pallas_call" in (FINAL / r["path"]).read_text() for r in recs if (FINAL / r["path"]).exists()):
        bad.append("a Pallas kernel is present in final/")
    return bad


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--val-frac", type=float, default=0.1); ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        bad = check()
        print("\n".join(bad) if bad else f"export check: ok ({len(read_jsonl(FINAL / 'problems.jsonl'))} problems)")
        sys.exit(1 if bad else 0)
    m = export(a.val_frac)
    print(f"exported {m['n_problems']} problems ({m['by_source']}, splits {m['by_split']}), {m['n_excluded']} excluded -> {FINAL}")


if __name__ == "__main__":
    main()
