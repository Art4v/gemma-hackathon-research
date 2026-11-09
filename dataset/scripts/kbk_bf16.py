"""Switch half of the KernelBook ports to bfloat16 inputs, in place. Every port defaults to float32 (PyTorch's default), while 29 of
JAXBench's 64 tasks and 55% of dataset-v1's kernels are bf16. A deterministic half (sha256 of the problem id, odd) gets
make_variants.py's own edit (create_inputs' default dtype switched, docstring tagged), recorded as source_ref.config_delta, with
the signature reset so filter_contamination.py recomputes it. Switching instead of adding a copy keeps the count and avoids
near-twin pairs. Run after porting finishes and before the filters; idempotent.
  python dataset/scripts/kbk_bf16.py [--batch kbk] [--all] [--dry-run]     (kbk: half, k-tasks are all f32; prod: --all, p-tasks 29/31 bf16)
"""
from __future__ import annotations

import argparse, hashlib, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import PROBLEMS, read_jsonl, sha256, write_jsonl  # noqa: E402
from make_variants import variant_code  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", default="kbk"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--all", action="store_true", help="switch every eligible problem (the prod batch: JAXBench's p-tasks are 29/31 bf16)")
    a = ap.parse_args(argv)
    rows, n, skipped = read_jsonl(PROBLEMS / "problems.jsonl"), 0, 0
    for r in rows:
        if r.get("batch") != a.batch or r["status"] != "ok" or (r["source_ref"].get("config_delta") or {}).get("dtype"):
            continue
        if not a.all and int(hashlib.sha256(r["problem_id"].encode()).hexdigest(), 16) % 2 == 0:
            continue
        code = variant_code(r["ref_code"])
        if code is None:
            skipped += 1; continue
        r.update(ref_code=code, ref_sha256=sha256(code), signature=None, input_bytes=None, fits_hbm=None,
                 source_ref={**r["source_ref"], "config_delta": {"dtype": "bfloat16"}})
        if not a.dry_run:
            (PROBLEMS / f"{r['problem_id']}.py").write_text(code)
        n += 1
    print(f"{n} switched to bf16, {skipped} skipped (create_inputs default not a single float32)")
    if not a.dry_run and n:
        write_jsonl(PROBLEMS / "problems.jsonl", rows)


if __name__ == "__main__":
    main()
