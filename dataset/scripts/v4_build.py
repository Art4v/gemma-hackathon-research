"""Dataset v4: the v3 train file that wins the val ablation, plus every TPU-verified row from dataset/v4/<part>.jsonl, in the training format.

  python dataset/scripts/v4_build.py --base train|train_capped [--coverage curation/coverage.py] [--python PY]

Rows from the parts keep kind single or repair, teacher claude, verification tpu. Val is dataset-v1's 26 rows, unchanged. Writes dataset/v4/final/
{train.jsonl, val.jsonl, verification.jsonl, MANIFEST.json} and, with --coverage, coverage.txt. Fails on a duplicate example id or a row from a val family.
"""
from __future__ import annotations

import argparse, collections, contextlib, io, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import FINAL, now, read_jsonl, sha256, write_json, write_jsonl  # noqa: E402
from v3_build import CHARS_PER_TOKEN, MAX_TARGET_TOKENS, bench, training_row  # noqa: E402
from v3_common import V3  # noqa: E402
from v4_parts import V4  # noqa: E402

PART_ORDER = ("m1", "kbk", "kbkc", "kbkd", "prod")
# A target longer than the eval's 8,192-token generation cap teaches output the eval would cut off. chars/2.44 ran ~5% low on the one real case
# (ConvDenoiser_3791: 19,347 chars, 8,316 tokens by Gemma's tokenizer), so the cap uses a stricter ratio: 2.3 chars per token, i.e. 18,000 chars.
MAX_TARGET_CHARS = 18000


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, choices=("train", "train_capped")); ap.add_argument("--coverage", default="")
    a = ap.parse_args(argv)
    conv = bench().conversation
    # families dropped by rule 3, from train and val alike; a training row's family is the first two parts of its task name
    excl = {l.strip() for l in (V4 / "exclude_families.txt").read_text().splitlines() if l.strip() and not l.startswith("#")}
    dropped_family = collections.Counter()

    def keep(task: str) -> bool:
        """An entry names a family (authored/attention_alibi) or a single problem (scraped/maxtext/ragged_attention_v2); it drops the task
        and anything under it, such as an m1 child (task name __-joined, so the path is the task with __ read as /)."""
        path = task.replace("__", "/")
        for e in excl:
            if path == e or path.startswith(e + "/"):
                dropped_family[e] += 1
                return False
        return True

    base = [r for r in read_jsonl(V3 / "final" / f"{a.base}.jsonl") if keep(r["task"])]
    base_verif = {v["example_id"]: v for v in read_jsonl(V3 / "final" / "verification.jsonl")}
    val_all = read_jsonl(V3 / "final" / "val.jsonl")
    val = [r for r in val_all if keep(r["task"])]
    val_families = {e["family_id"] for e in read_jsonl(FINAL / "val.jsonl")}
    rows, verif, parts, too_long = list(base), [base_verif[r["example_id"]] for r in base if r["example_id"] in base_verif], {}, []
    for name in PART_ORDER:
        part = read_jsonl(V4 / f"{name}.jsonl")
        parts[name] = len(part)
        for e in part:
            if e["family_id"] in val_families:
                sys.exit(f"{e['example_id']}: from a val family")
            if not keep(e["task_name"]):  # by task name, like the base rows: m1 children of an excluded problem go too
                continue
            if len(e["turns"][-1]["code"]) > MAX_TARGET_CHARS:
                too_long.append(e["example_id"])
                continue
            rows.append(training_row(e, e["kind"], conv))
            verif.append({"example_id": e["example_id"], "kind": e["kind"], "part": name, "verification": "tpu", "final_verdict": e["final_verdict"],
                          "speed_vs_xla": e["speed_vs_xla"], "oracle_job_ids": e["oracle"]["job_ids"], "oracle_shas": e["oracle"]["shas"]})
    ids = collections.Counter(r["example_id"] for r in rows + val)
    dup = [i for i, n in ids.items() if n > 1]
    if dup:
        sys.exit(f"{len(dup)} duplicate example ids, e.g. {dup[:3]}")
    out = V4 / "final"
    write_jsonl(out / "train.jsonl", rows)
    write_jsonl(out / "val.jsonl", val)
    write_jsonl(out / "verification.jsonl", verif)
    files = ["train.jsonl", "val.jsonl", "verification.jsonl"]
    if a.coverage:
        import importlib.util
        spec = importlib.util.spec_from_file_location("coverage_v4", a.coverage)
        cov = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cov)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cov.report(str(out / "train.jsonl"))
        (out / "coverage.txt").write_text(buf.getvalue())
        files.append("coverage.txt")
    lens = [len(r["messages"][-1]["content"]) for r in rows]
    total = [sum(len(m["content"]) for m in r["messages"]) for r in rows]
    parts_report = json.loads((V4 / "parts_report.json").read_text()) if (V4 / "parts_report.json").exists() else {}
    manifest = {"created_at": now(), "base": f"v3 {a.base}", "n_train": len(rows), "n_val": len(val), "base_rows": len(base), "part_rows": parts, "dropped_target_over_cap": {"max_chars": MAX_TARGET_CHARS, "ids": too_long},
                "excluded_families": {"file": "dataset/v4/exclude_families.txt", "families": sorted(excl), "rows_dropped_train_and_val": dict(sorted(dropped_family.items())),
                                      "val_rows_before": len(val_all), "val_problems": len({r["task"] for r in val})},
                "part_commits": {k: v.get("commit") for k, v in parts_report.items() if isinstance(v, dict) and "commit" in v},
                "train_by_kind": dict(collections.Counter(r["kind"] for r in rows)), "by_teacher": dict(collections.Counter(r["teacher"] for r in rows)),
                "by_verification": dict(collections.Counter(v["verification"] for v in verif)),
                "tokens_ESTIMATED": {"how": f"chars / {CHARS_PER_TOKEN}; confirm with build_rows' tokenizer", "max_target": int(max(lens) / CHARS_PER_TOKEN),
                                     "max_row": int(max(total) / CHARS_PER_TOKEN), "max_target_limit": MAX_TARGET_TOKENS},
                "files": {f: sha256((out / f).read_bytes()) for f in files}}
    write_json(out / "MANIFEST.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("base", "n_train", "n_val", "part_rows", "train_by_kind", "by_verification", "tokens_ESTIMATED")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
