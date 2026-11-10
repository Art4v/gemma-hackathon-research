"""Per-step VMEM block bytes of each shape variant's kernel, at the variant's shape and at the original (TPU-passed) shape.

  python dataset/scripts/v3_block_bytes.py [--python PY]

Traces each kernel on abstract inputs (nothing runs) and sums, over every pallas_call, the bytes of one block per operand and output
(block shape x dtype size; a squeezed None dim counts as 1), doubled for the pipeline's double buffering. Writes dataset/v4/shape_variant_blocks.jsonl.
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
    for k in ("variant", "original"):
        try:
            r[k] = block_bytes(it[k + "_ref"], it["code"])
        except Exception as e:
            r[k] = None; r[k + "_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    print("@@" + json.dumps(r), flush=True)
'''


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--python", default=None)
    a = ap.parse_args(argv)
    tr = {json.loads(l)["example_id"] for l in (V3.parent / "v4" / "final" / "train.jsonl").read_text().splitlines() if l.strip()}
    base = {e["example_id"]: e for e in read_jsonl(V3.parent / "final" / "train.jsonl")}
    items = []
    for v in read_jsonl(V3 / "shape_variants.jsonl"):
        if v["example_id"] not in tr:
            continue
        orig = base[v["derived_from"]]
        items.append({"id": v["example_id"], "code": v["turns"][-1]["code"], "variant_ref": v["turns"][0]["ref_code"], "original_ref": orig["turns"][0]["ref_code"]})
    files = {p: (ORACLE / p).read_text() for p in ORACLE_FILES}
    files.update({"child.py": CHILD, "items.json": json.dumps(items)})
    r = Isolated(a.python, timeout=1800, max_rss_gb=6).run(["child.py"], files=files, extra_env={"ORACLE_COMMIT": "v3-blocks"})
    rows = [json.loads(l[2:]) for l in r.stdout.splitlines() if l.startswith("@@")]
    if len(rows) < len(items):
        print("child stopped early:", (r.stderr.strip().splitlines() or ["?"])[-1][:300])
    write_jsonl(V3.parent / "v4" / "shape_variant_blocks.jsonl", rows)
    ok = [x for x in rows if x["variant"] is not None and x["original"] is not None]
    grew = [x for x in ok if x["variant"] > x["original"]]
    mib = lambda b: round(b / 2 ** 20, 2)
    print(f"{len(rows)} of {len(items)} traced; both shapes measured {len(ok)}; variant larger than original {len(grew)}")
    print("max variant MiB (x2 buffered):", max((mib(x["variant"]) for x in ok), default=None), " max original:", max((mib(x["original"]) for x in ok), default=None))
    for x in grew[:10]:
        print("  GREW", x["id"], mib(x["original"]), "->", mib(x["variant"]))
    for x in [x for x in rows if x["variant"] is None or x["original"] is None][:5]:
        print("  ERR", x["id"], x.get("variant_error") or x.get("original_error"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
