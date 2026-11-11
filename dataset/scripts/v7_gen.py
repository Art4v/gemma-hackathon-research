"""v7: exam-scale (reference, Pallas kernel) pairs from v3_generate's matmul templates, under one tiling policy, for the TPU oracle.

  python dataset/scripts/v7_gen.py [--n 300] [--seed 0] [--python PY] [--max-mib 90]

Families (the two designs v4 lacks on the matmul k-tasks, FINETUNE_ANALYSIS.md section 3.2):
  matmul_epilogue, bmm_epilogue  K tiled with an f32 accumulator in scratch, the epilogue on the last K step (rule A)
  row_reduction_matmul           the weight block spans all of N, so the row reduction sees whole rows (rule B)
Sizes: K and N in {4096, 8192}, M in {2048, 4096} for matmul and bmm ({1024, 2048, 4096} for row reductions), weight >= 64 MB, inputs <= 800 MB. The kernels pick their tiles at trace time from one policy
(_tile(n, pref): the largest power of two <= pref dividing n) and set one vmem_limit_bytes, so the same text runs at every size.

Pipeline, all CPU, writes dataset/v7_gen/:
  1. candidates.jsonl         each with a full-size reference and a shrunken one (same kernel; M and N cut, K kept so K still tiles)
  2. full_verified.jsonl      v3_verify at full size without numerics: reference checks, signature, static step 1, v6e lowering
  3. filter/                  filter_contamination (L1-L2) and filter_neardup (L3) on the full-size references, as v3 ran them
  4. shrunk_verified.jsonl    v3_verify on the shrunken reference with numerics (interpret mode, 3 seeds)
  5. blocks.jsonl             v3_block_bytes at full size: 2 x blocks + scratch, against the kernel's own limit
  6. items.jsonl              survivors in the kimi/v7 item format (set e5_gen), for the TPU oracle; nothing here has been on a chip
"""
from __future__ import annotations

import argparse, collections, pathlib, random, subprocess, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import read_jsonl, write_jsonl  # noqa: E402
from v3_block_bytes import measure  # noqa: E402
from v3_common import DATASET  # noqa: E402
from v3_generate import HEADER_K, REDUCE_KEEP, _chain_names, apply_lines, epilogue, ref_module  # noqa: E402
from v3_verify import verify  # noqa: E402

OUT = DATASET / "v7_gen"
# Policy from three v6e sweeps on Karthik's project (#5, Nov 11, oracle c9a07d9298a1). One limit for every kernel: 124 MiB, because the
# f32 row reduction at N = 8192 only clears the bar with tm = 1024 (114 MiB est., 1.01x); 100 MiB caps it at 0.58x.
VMEM_LIMIT_MIB = 124
MATMUL_TILES = (2048, 2048, 1024)   # (tm, tn, tk): f32 0.87x at M 2048, 1.31x at M 4096; 2 x (8 + 8 + 16) + 16 acc = 80 MiB


def rowred_prefs(n: int) -> dict:
    """Row reduction (tm, tk) by N: N 4096 -> (1024, 512), 1.23x at 68 MiB; N 8192 -> (1024, 256), 1.01x at 114 MiB. tk never below 128."""
    return {"m": 1024, "k": 512 if n <= 4096 else 256}
# CPU numerics run at a shrunken CONFIG: each axis that is tiled at full size keeps exactly 2 tiles (dim = 2 x preferred tile), so the
# accumulator reset, the last-step epilogue and every index map are exercised across a tile boundary (the v4 `wrong` bugs only show then).


def _tile(n: int, pref: int) -> int:
    """Same rule as the generated kernels' _tile."""
    t = min(pref, n)
    while n % t:
        t //= 2
    return t


def _shrink(full: dict, prefs: dict) -> dict:
    """{axis: dim} at the shrunken size: an axis tiled at full size gets 2 x its preferred tile (never more than its full size)."""
    out = {}
    for ax, n in full.items():
        p = prefs.get(ax)
        out[ax] = n if p is None or n <= p else min(n, 2 * p)
    return out


def _check_steps(full: dict, small: dict, prefs: dict) -> None:
    for ax, p in prefs.items():
        if full[ax] // _tile(full[ax], p) >= 2:
            assert small[ax] // _tile(small[ax], p) >= 2, f"axis {ax}: {small[ax]} gives one tile of {_tile(small[ax], p)} at the shrunken size"

POLICY = (f"\nVMEM_LIMIT = {VMEM_LIMIT_MIB} * 1024 * 1024  # one scoped VMEM limit for every kernel (v6e has 128 MiB of VMEM)\n\n\n"
          "def _tile(n, pref):\n    \"\"\"The largest power-of-two block <= pref that divides n (n itself when n <= pref).\"\"\"\n"
          "    t = min(pref, n)\n    while n % t:\n        t //= 2\n    return t\n")


def _dtype(rng) -> tuple:
    return ("jnp.bfloat16", 2) if rng.random() < 0.4 else ("jnp.float32", 4)


def _sizes(rng, nbytes: int, batched: bool, ms: tuple = (1024, 2048, 4096)) -> tuple:
    """Inputs capped at 800 MB: larger ones overran the 6 GB host-memory cap of the CPU check (9 of 300 in the first run)."""
    while True:
        K, N = rng.choice((4096, 8192)), rng.choice((4096, 8192))
        M = rng.choice(ms)
        B = rng.choice((2, 4)) if batched else 1
        if K * N * nbytes >= 64 * 2 ** 20 and B * (M * K + K * N + M * N) * nbytes <= 800 * 2 ** 20:
            return B, M, N, K


def _dot(dtype: str) -> str:
    # bf16 operands go to the MXU as they are; f32 operands as in v3 (exact products, f32 accumulate)
    cast = "" if dtype == "jnp.bfloat16" else ".astype(jnp.float32)"
    return f"jnp.dot(a_ref[...]{cast}, b_ref[...]{cast}, preferred_element_type=jnp.float32)"


def matmul_epilogue(i: int, rng) -> dict:
    batched = rng.random() < 0.25
    dtype, nb = _dtype(rng)
    B, M, N, K = _sizes(rng, nb, batched, ms=(2048, 4096))  # the sweep: an output tile under 2048 x 2048 stays below 0.7x on f32
    bias = rng.random() < 0.3
    chain = epilogue(rng)
    pre = "c['batch'], " if batched else ""
    inputs = [("a", f"({pre}c['m'], c['k'])", 0.05), ("b", f"({pre}c['k'], c['n'])", 0.05)] + ([("v", "(c['n'],)", 0.1)] if bias else [])
    sig = "a, b" + (", v" if bias else "")
    body = "    y = jnp.matmul(a, b)\n" + ("    y = y + v\n" if bias else "") + apply_lines(chain, "    ") + "    return y\n"
    doc = f"{'Batched matmul' if batched else 'Matmul'}{' -> bias' if bias else ''} -> {_chain_names(chain)}."
    name = f"gen7_{'bmm' if batched else 'matmul'}_{i}"
    cfg = {"name": name, **({"batch": B} if batched else {}), "m": M, "n": N, "k": K}
    prefs = dict(zip("mnk", MATMUL_TILES))
    dims = _shrink({"m": M, "n": N, "k": K}, prefs)
    _check_steps({"m": M, "n": N, "k": K}, dims, prefs)
    small = {**cfg, **dims, **({"batch": min(B, 2)} if batched else {})}
    g = (["b_i"] if batched else []) + ["i", "j", "l"]
    args, kn = ", ".join(g), len(g) - 1
    bdim, bix = ("None, ", "b_i, ") if batched else ("", "")
    kern = (f"\n\ndef _kernel(a_ref, b_ref{', v_ref' if bias else ''}, o_ref, acc):\n    @pl.when(pl.program_id({kn}) == 0)\n    def _():\n        acc[...] = jnp.zeros_like(acc)\n"
            f"    acc[...] += {_dot(dtype)}\n\n"
            f"    @pl.when(pl.program_id({kn}) == pl.num_programs({kn}) - 1)\n    def _():\n        y = acc[...]\n"
            + ("        y = y + v_ref[...].astype(jnp.float32)\n" if bias else "") + apply_lines(chain, "        ") + "        o_ref[...] = y.astype(o_ref.dtype)\n")
    head = ("    B, M, K = a.shape\n    N = b.shape[2]\n" if batched else "    M, K = a.shape\n    N = b.shape[1]\n")
    tm, tn, tk = MATMUL_TILES
    head += f"    tm, tn, tk = _tile(M, {tm}), _tile(N, {tn}), _tile(K, {tk})\n"
    grid = "(B, M // tm, N // tn, K // tk)" if batched else "(M // tm, N // tn, K // tk)"
    sem = ", ".join(['"parallel"'] * (len(g) - 1) + ['"arbitrary"'])
    vspec = f", pl.BlockSpec((tn,), lambda {args}: (j,))" if bias else ""
    code = (f"{HEADER_K}{POLICY}{kern}\n\ndef workload({sig}):\n{head}    return pl.pallas_call(\n        _kernel,\n"
            f"        out_shape=jax.ShapeDtypeStruct(({'B, ' if batched else ''}M, N), a.dtype),\n        grid={grid},\n"
            f"        in_specs=[pl.BlockSpec(({bdim}tm, tk), lambda {args}: ({bix}i, l)), pl.BlockSpec(({bdim}tk, tn), lambda {args}: ({bix}l, j)){vspec}],\n"
            f"        out_specs=pl.BlockSpec(({bdim}tm, tn), lambda {args}: ({bix}i, j)),\n        scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],\n"
            f"        compiler_params=pltpu.CompilerParams(dimension_semantics=({sem}), vmem_limit_bytes=VMEM_LIMIT),\n    )({sig})\n")
    return {"family": "bmm_epilogue" if batched else "matmul_epilogue", "name": name, "code": code,
            "ref_code": ref_module(doc, cfg, dtype, inputs, sig, body), "shrunk_ref_code": ref_module(doc, small, dtype, inputs, sig, body)}


def row_reduction_matmul(i: int, rng) -> dict:
    dtype, nb = _dtype(rng)
    _, M, N, K = _sizes(rng, nb, False)
    pre, post = epilogue(rng, 0, 2), epilogue(rng, 0, 1)
    rname = rng.choice(sorted(REDUCE_KEEP))
    keeps = REDUCE_KEEP[rname].startswith("jax.nn") or rname in ("normalize", "center")
    Nout = "N" if keeps else "1"
    chain = pre + [(rname, REDUCE_KEEP[rname])] + post
    inputs, sig = [("a", "(c['m'], c['k'])", 0.05), ("b", "(c['k'], c['n'])", 0.05)], "a, b"
    body = "    y = jnp.matmul(a, b)\n" + apply_lines(chain, "    ") + "    return y\n"
    doc = f"Matmul -> {_chain_names(chain)} over the last axis."
    name = f"gen7_rowreduce_{i}"
    cfg = {"name": name, "m": M, "n": N, "k": K}
    sn = min(N, 2048)  # N stays wider than any plausible N tile, so a kernel reducing over part of a row fails here
    dims = _shrink({"m": M, "k": K}, rowred_prefs(sn))  # the shrunken run uses the prefs the kernel will pick at the shrunken N
    _check_steps({"m": M, "k": K}, dims, rowred_prefs(sn))
    small = {**cfg, **dims, "n": sn}
    kern = ("\n\ndef _kernel(a_ref, b_ref, o_ref, acc):\n    @pl.when(pl.program_id(1) == 0)\n    def _():\n        acc[...] = jnp.zeros_like(acc)\n"
            f"    acc[...] += {_dot(dtype)}\n\n"
            "    @pl.when(pl.program_id(1) == pl.num_programs(1) - 1)\n    def _():\n        y = acc[...]\n" + apply_lines(chain, "        ") + "        o_ref[...] = y.astype(o_ref.dtype)\n")
    code = (f"{HEADER_K}{POLICY}{kern}\n\ndef workload(a, b):\n    M, K = a.shape\n    N = b.shape[1]\n"
            f"    # rows per block decide the speed; at N = 8192 the whole-row blocks need a smaller K tile to fit\n"
            f"    tm, tk = _tile(M, 1024), _tile(K, 512 if N <= 4096 else 256)\n"
            f"    # the weight block spans all of N: the row reduction must see whole rows\n"
            f"    return pl.pallas_call(\n        _kernel,\n        out_shape=jax.ShapeDtypeStruct((M, {Nout}), a.dtype),\n        grid=(M // tm, K // tk),\n"
            f"        in_specs=[pl.BlockSpec((tm, tk), lambda i, l: (i, l)), pl.BlockSpec((tk, N), lambda i, l: (l, 0))],\n"
            f"        out_specs=pl.BlockSpec((tm, {Nout}), lambda i, l: (i, 0)),\n        scratch_shapes=[pltpu.VMEM((tm, N), jnp.float32)],\n"
            '        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary"), vmem_limit_bytes=VMEM_LIMIT),\n    )(a, b)\n')
    return {"family": "row_reduction_matmul", "name": name, "code": code,
            "ref_code": ref_module(doc, cfg, dtype, inputs, sig, body), "shrunk_ref_code": ref_module(doc, small, dtype, inputs, sig, body)}


FAMILIES = [(matmul_epilogue, 0.62), (row_reduction_matmul, 0.38)]  # bmm is 25% of matmul_epilogue; row reductions weighted up (rule B is the smallest group)


def candidates(n: int, seed: int) -> list:
    out = []
    for f, share in FAMILIES:
        for k in range(round(n * share)):
            c = f(k, random.Random(f"v7/{seed}/{f.__name__}/{k}"))
            cid = f"gen7__{f.__name__}__{k:05d}"
            out.append({"id": cid, "task": cid, **c})
    return out


def run_filter(script: str, problems: pathlib.Path, python: str | None) -> None:
    cmd = [sys.executable, "-I", str(pathlib.Path(__file__).resolve().parent / script), "--problems", str(problems)] + (["--python", python] if python and script == "filter_contamination.py" else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = [l for l in r.stdout.splitlines() if l.startswith("kept")]
    print(f"  {script}: {tail[-1] if tail else (r.stderr.strip().splitlines() or ['?'])[-1][:200]}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--python", default=None); ap.add_argument("--max-mib", type=float, default=116.0, help="drop a kernel whose estimated VMEM exceeds this (114 MiB ran clean against the 124 MiB limit)")
    a = ap.parse_args(argv)
    log = lambda m: print(m, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cands = candidates(a.n, a.seed)
    write_jsonl(OUT / "candidates.jsonl", cands)
    log(f"{len(cands)} candidates: {dict(collections.Counter(c['family'] for c in cands))}")

    full = verify([{"id": c["id"], "ref_code": c["ref_code"], "code": c["code"]} for c in cands], OUT / "full_verified.jsonl", a.python,
                  budget=180, numerics=False, log=log)
    ok = [c for c in cands if full[c["id"]].get("ok")]
    log(f"full size (ref, static, v6e lowering): {len(ok)} of {len(cands)}; failures {dict(collections.Counter(full[c['id']].get('stage') for c in cands if not full[c['id']].get('ok')))}")

    fdir = OUT / "filter"; fdir.mkdir(exist_ok=True)
    for p in ("excluded.jsonl", "review.jsonl", "signatures.jsonl"):
        (fdir / p).unlink(missing_ok=True)
    write_jsonl(fdir / "problems.jsonl", [{"problem_id": c["id"], "status": "ok", "source": "generated", "ref_code": c["ref_code"],
                                            "signature": full[c["id"]]["signature"]} for c in ok])
    run_filter("filter_contamination.py", fdir / "problems.jsonl", a.python)
    run_filter("filter_neardup.py", fdir / "problems.jsonl", a.python)
    kept = {r["problem_id"] for r in read_jsonl(fdir / "problems.jsonl") if r["status"] == "ok"}
    ok = [c for c in ok if c["id"] in kept]
    log(f"after contamination filters: {len(ok)}")

    small = verify([{"id": c["id"], "ref_code": c["shrunk_ref_code"], "code": c["code"]} for c in ok], OUT / "shrunk_verified.jsonl", a.python, budget=180, log=log)
    ok = [c for c in ok if small[c["id"]].get("ok")]
    log(f"numerics at the shrunken size: {len(ok)}")

    blocks = {r["id"]: r for r in measure([{"id": c["id"], "code": c["code"], "refs": {"full": c["ref_code"]}} for c in ok], a.python)}
    write_jsonl(OUT / "blocks.jsonl", list(blocks.values()))
    cap = a.max_mib * 2 ** 20
    fits = [c for c in ok if (blocks.get(c["id"]) or {}).get("full_total") is not None and blocks[c["id"]]["full_total"] <= cap]
    log(f"estimated VMEM <= {a.max_mib} MiB: {len(fits)} of {len(ok)}")

    items = [{"id": f"gen7/{c['id']}", "task": c["task"], "set": "e5_gen", "family": c["family"], "ref_code": c["ref_code"], "code": c["code"],
              "vmem_estimate_mib": round(blocks[c["id"]]["full_total"] / 2 ** 20, 2), "verification": "cpu (full-size lowering, shrunken numerics)"} for c in fits]
    write_jsonl(OUT / "items.jsonl", items)
    log(f"items.jsonl: {len(items)} {dict(collections.Counter(i['family'] for i in items))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
