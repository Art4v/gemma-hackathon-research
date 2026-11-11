"""Per-step VMEM block bytes of Pallas kernels, traced on abstract inputs (nothing runs, CPU only).

  python dataset/scripts/v3_block_bytes.py [--python PY]
      the 66 v4 shape variants, at the variant's shape and at the original (TPU-passed) shape -> dataset/v4/shape_variant_blocks.jsonl
  python dataset/scripts/v3_block_bytes.py --items ITEMS.jsonl --out OUT.jsonl [--python PY]
      any kernels: one JSON object per line, {"id": ..., "code": <kernel module>, "refs": {<label>: <reference module>, ...}};
      writes {"id", <label>: bytes or null, <label>_error?} per item, one value per label (e.g. {"small": ..., "scaled": ...})

For each pallas_call, sums the bytes of one block per operand and output (block shape x dtype size; a squeezed None dim counts as 1);
takes the largest pallas_call; doubles it for the pipeline's double buffering. Not counted: scratch_shapes (accumulators) and in-kernel
temporaries, so the true footprint is at least this. Compare against the scoped VMEM limit the kernel sets (v6e default 32 MiB).
"""
from __future__ import annotations

import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import ORACLE, Isolated, read_jsonl, write_jsonl  # noqa: E402
from v3_common import ORACLE_FILES, V3  # noqa: E402

CHILD = r'''
import json, sys, tempfile
sys.path.insert(0, ".")
import tpu_oracle as o
o.init(cpu_smoke=True)
jax, np, tc = o.jax, o.np, o.tc

def walk(jaxpr, out):
    for e in jaxpr.eqns:
        if e.primitive.name == "pallas_call":
            gm = e.params["grid_mapping"]
            total = 0
            for bm in gm.block_mappings:
                shape = [1 if d is None or not isinstance(d, int) else d for d in getattr(bm, "block_shape", ())]
                # block_shape entries may be Blocked(n)/Squeezed objects in newer JAX
                shape = [getattr(d, "block_size", d) if not isinstance(d, int) else d for d in getattr(bm, "block_shape", ())]
                shape = [d if isinstance(d, int) else 1 for d in shape]
                n = 1
                for d in shape:
                    n *= d
                total += n * np.dtype(bm.array_aval.dtype).itemsize
            out.append(total)
        for p in e.params.values():
            for s in (p if isinstance(p, (list, tuple)) else [p]):
                j = getattr(s, "jaxpr", None)
                if j is not None and hasattr(j, "eqns"):
                    walk(j, out)
                elif hasattr(s, "eqns"):
                    walk(s, out)

def block_bytes(ref, code):
    t = o.Task("task", ref); t.load()
    with tempfile.TemporaryDirectory() as d:
        cand = tc.load(code, "cand", d)
        jp = jax.make_jaxpr(cand.workload, static_argnums=t.static)(*t.abstract).jaxpr
    out = []
    walk(jp, out)
    return 2 * max(out) if out else None

for it in json.load(open("items.json")):
    r = {"id": it["id"]}
    for k, ref in it["refs"].items():
        try:
            r[k] = block_bytes(ref, it["code"])
        except Exception as e:
            r[k] = None; r[k + "_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    print("@@" + json.dumps(r), flush=True)
'''


def measure(items: list, python: str | None, timeout: int = 1800) -> list:
    """Rows for `items` ({"id", "code", "refs": {label: ref}}), traced in one isolated child; may be shorter if the child dies."""
    files = {p: (ORACLE / p).read_text() for p in ORACLE_FILES}
    files.update({"child.py": CHILD, "items.json": json.dumps(items)})
    r = Isolated(python, timeout=timeout, max_rss_gb=6).run(["child.py"], files=files, extra_env={"ORACLE_COMMIT": "v3-blocks"})
    rows = [json.loads(l[2:]) for l in r.stdout.splitlines() if l.startswith("@@")]
    if len(rows) < len(items):
        print("child stopped early:", (r.stderr.strip().splitlines() or ["?"])[-1][:300])
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0]); ap.add_argument("--python", default=None)
    ap.add_argument("--items", default=None, help="JSONL of {id, code, refs: {label: ref}}; measure these instead of the v4 shape variants")
    ap.add_argument("--out", default=None, help="output JSONL (required with --items)")
    a = ap.parse_args(argv)
    mib = lambda b: round(b / 2 ** 20, 2)
    if a.items:
        if not a.out:
            ap.error("--out is required with --items")
        items = read_jsonl(pathlib.Path(a.items))
        rows = measure(items, a.python, timeout=max(1800, 10 * len(items)))
        write_jsonl(pathlib.Path(a.out), rows)
        labels = sorted({k for it in items for k in it["refs"]})
        print(f"{len(rows)} of {len(items)} traced -> {a.out}")
        for k in labels:
            ok = [x[k] for x in rows if x.get(k) is not None]
            over = sum(b > 32 * 2 ** 20 for b in ok)
            print(f"  {k}: measured {len(ok)}, max {mib(max(ok)) if ok else None} MiB, over 32 MiB {over}, errors {sum(x.get(k) is None for x in rows)}")
        return 0
    tr ={json.loads(l)["example_id"] for l in (V3.parent / "v4" / "final" / "train.jsonl").read_text().splitlines() if l.strip()}
    base = {e["example_id"]: e for e in read_jsonl(V3.parent / "final" / "train.jsonl")}
    items = []
    for v in read_jsonl(V3 / "shape_variants.jsonl"):
        if v["example_id"] not in tr:
            continue
        orig = base[v["derived_from"]]
        items.append({"id": v["example_id"], "code": v["turns"][-1]["code"],
                      "refs": {"variant": v["turns"][0]["ref_code"], "original": orig["turns"][0]["ref_code"]}})
    rows = measure(items, a.python)
    write_jsonl(V3.parent / "v4" / "shape_variant_blocks.jsonl", rows)
    ok = [x for x in rows if x["variant"] is not None and x["original"] is not None]
    grew = [x for x in ok if x["variant"] > x["original"]]
    print(f"{len(rows)} of {len(items)} traced; both shapes measured {len(ok)}; variant larger than original {len(grew)}")
    print("max variant MiB (x2 buffered):", max((mib(x["variant"]) for x in ok), default=None), " max original:", max((mib(x["original"]) for x in ok), default=None))
    for x in grew[:10]:
        print("  GREW", x["id"], mib(x["original"]), "->", mib(x["variant"]))
    for x in [x for x in rows if x["variant"] is None or x["original"] is None][:5]:
        print("  ERR", x["id"], x.get("variant_error") or x.get("original_error"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
