"""Dataset v3: drop generated elementwise and reduction rows with 8 MiB or more of input blocks per grid step (Hruday's decision, issue 5, Nov 9).

  python dataset/scripts/v3_drop_large_blocks.py

The CPU lowering cannot see VMEM. On oracle-results/v3-gen-sample (100 generated rows on a real v6e) the elementwise and reduction rows split by this measure: under 8 MiB
30 pass, 0 vmem; at or above it 5 pass, 10 vmem, 1 correct_slow. Other families are untouched: all of them passed in that sample. Rewrites generated.jsonl, moves the
dropped rows to dropped_large_blocks.jsonl. Run v3_build.py and v3_capped.py after it.
"""
from __future__ import annotations

import collections, pathlib, re, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import now, read_jsonl, write_json, write_jsonl  # noqa: E402
from v3_common import V3  # noqa: E402

LIMIT = 8 * 1024 * 1024
FAMILIES = ("elementwise", "reduction")


def input_block_bytes(row: dict) -> int:
    """The sum over in_specs of block elements times the element size of the reference's input dtype."""
    code, ref = row["turns"][1]["code"], row["turns"][0]["ref_code"]
    specs = code.split("in_specs=", 1)[1].split("out_specs", 1)[0]
    size = 2 if "bfloat16" in ref.split("def create_inputs", 1)[1][:40] else 4
    total = 0
    for m in re.finditer(r"pl\.BlockSpec\(\(([^)]*)\)", specs):
        n = 1
        for x in m.group(1).split(","):
            if x.strip().isdigit():
                n *= int(x)
        total += n * size
    return total


def main() -> int:
    rows = read_jsonl(V3 / "generated.jsonl")
    keep, drop = [], []
    for r in rows:
        (drop if r["meta"]["v3"]["family"] in FAMILIES and input_block_bytes(r) >= LIMIT else keep).append(r)
    write_jsonl(V3 / "generated.jsonl", keep)
    write_jsonl(V3 / "dropped_large_blocks.jsonl", drop, append=(V3 / "dropped_large_blocks.jsonl").exists())
    write_json(V3 / "large_blocks_report.json", {"at": now(), "limit_bytes": LIMIT, "families": FAMILIES, "n_before": len(rows), "n_kept": len(keep), "n_dropped": len(drop),
                                                 "dropped_by_family": dict(collections.Counter(r["meta"]["v3"]["family"] for r in drop)),
                                                 "kept_by_family": dict(collections.Counter(r["meta"]["v3"]["family"] for r in keep)), "evidence": "oracle-results/v3-gen-sample"})
    print(f"kept {len(keep)} of {len(rows)}; dropped {len(drop)}: {dict(collections.Counter(r['meta']['v3']['family'] for r in drop))}")
    print(dict(collections.Counter(r["meta"]["v3"]["family"] for r in keep)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
