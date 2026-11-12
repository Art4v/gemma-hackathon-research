"""Dataset v3: a second train file with generated rows capped per structural skeleton, so that the generator cannot crowd out dataset-v1's API variety.

  python dataset/scripts/v3_capped.py --coverage curation/coverage.py [--top10-max 0.30]

Skeletons are coverage.py's (Karthik's agent, branch karthik/coverage): grid ranks, index-map arities, outputs, pallas_call count, feature flags. Takes the largest cap
per skeleton for which the top 10 skeletons hold at most --top10-max of the kernels coverage.py counts. Only generated rows are capped; rows are chosen by a stable hash.
Writes final/train_capped.jsonl, final/coverage.txt, and records both in MANIFEST.json. train.jsonl is left as it is.
"""
from __future__ import annotations

import argparse, collections, contextlib, importlib.util, io, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import read_json, read_jsonl, sha256, write_json, write_jsonl  # noqa: E402
from v3_common import V3  # noqa: E402

COUNTED = ("single", "generated", "shape_variant", "correct_slow", "cpu")  # the kinds coverage.report counts


def load(path: str):
    spec = importlib.util.spec_from_file_location("coverage_v3", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def top10_share(rows: list, cov) -> tuple:
    sk = collections.Counter(cov.skeleton(f) for f in (cov.features(cov.code_of(r)) for r in rows if r["kind"] in COUNTED) if f)
    n = sum(sk.values())
    return sum(c for _, c in sk.most_common(10)) / max(1, n), len(sk), n


def capped(rows: list, cov, cap: int) -> list:
    seen, out = collections.Counter(), []
    order = sorted(range(len(rows)), key=lambda i: sha256(rows[i]["example_id"]))
    drop = set()
    for i in order:
        if rows[i]["kind"] != "generated":
            continue
        f = cov.features(cov.code_of(rows[i]))
        k = cov.skeleton(f)
        seen[k] += 1
        if seen[k] > cap:
            drop.add(i)
    return [r for i, r in enumerate(rows) if i not in drop]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--coverage", required=True); ap.add_argument("--top10-max", type=float, default=0.30)
    a = ap.parse_args(argv)
    cov = load(a.coverage)
    out = V3 / "final"
    rows = read_jsonl(out / "train.jsonl")
    best, table = None, []
    for cap in range(1, 401):
        c = capped(rows, cov, cap)
        share, n_sk, n = top10_share(c, cov)
        table.append((cap, sum(r["kind"] == "generated" for r in c), round(share, 4), n_sk))
        if share <= a.top10_max:
            best = (cap, c, share, n_sk, n)
        else:
            break
    if best is None:
        sys.exit(f"no cap reaches top-10 share <= {a.top10_max}: {table[:3]}")
    cap, kept, share, n_sk, n = best
    write_jsonl(out / "train_capped.jsonl", kept)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cov.report(str(out / "train.jsonl")); cov.report(str(out / "train_capped.jsonl"))
    (out / "coverage.txt").write_text(buf.getvalue())
    m = read_json(out / "MANIFEST.json")
    m["capped"] = {"file": "train_capped.jsonl", "skeletons_from": "curation/coverage.py on karthik/coverage", "cap_per_skeleton_generated_rows": cap, "top10_share": round(share, 4), "top10_max": a.top10_max,
                   "skeletons_distinct": n_sk, "kernels_counted": n, "n_rows": len(kept), "train_by_kind": dict(collections.Counter(r["kind"] for r in kept))}
    m["files"]["train_capped.jsonl"] = sha256((out / "train_capped.jsonl").read_bytes())
    m["files"]["coverage.txt"] = sha256((out / "coverage.txt").read_bytes())
    write_json(out / "MANIFEST.json", m)
    print(f"cap {cap} per skeleton: {sum(r['kind'] == 'generated' for r in kept)} generated rows, {len(kept)} train rows, top-10 share {share:.3f}, {n_sk} skeletons; first caps {table[:6]}")
    print(buf.getvalue())
    return 0


if __name__ == "__main__":
    sys.exit(main())
