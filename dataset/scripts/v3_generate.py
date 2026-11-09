"""Dataset v3, part 2: a template generator of (reference, Pallas kernel) pairs, in one code style. Written from scratch: none of JAX's bundled
Pallas ops are used. Families: matmul and bmm with a random elementwise epilogue, matmul or a plain array followed by a row reduction, elementwise
ops and reductions across ranks, and convolution via im2col slicing and a Pallas matmul.

  python dataset/scripts/v3_generate.py [--n 2400] [--seed 0] [--limit-per-family K] [--python PY]

Every candidate goes through v3_verify.py; dataset/v3/generated_candidates.jsonl holds all of them and generated_verified.jsonl the results.
"""
from __future__ import annotations

import argparse, collections, pathlib, random, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import write_jsonl  # noqa: E402
from v3_common import V3  # noqa: E402
from v3_verify import verify  # noqa: E402

HEADER_K = "import jax\nimport jax.numpy as jnp\nfrom jax.experimental import pallas as pl\nfrom jax.experimental.pallas import tpu as pltpu\n"

# name -> (a label for the docstring, the expression on y); {c} and {d} are random constants filled in per use
ELEMENTWISE = {
    "relu": "jax.nn.relu(y)", "leaky_relu": "jax.nn.leaky_relu(y, {a})", "gelu": "jax.nn.gelu(y, approximate=True)", "sigmoid": "jax.nn.sigmoid(y)",
    "tanh": "jnp.tanh(y)", "swish": "y * jax.nn.sigmoid(y)", "mish": "y * jnp.tanh(jax.nn.softplus(y))", "hardtanh": "jnp.clip(y, -{b}, {b})",
    "elu": "jnp.where(y > 0, y, jnp.exp(y) - 1.0)", "softplus": "jax.nn.softplus(y)", "hard_swish": "y * jnp.clip(y / 6.0 + 0.5, 0.0, 1.0)",
    "hard_sigmoid": "jnp.clip(y / 6.0 + 0.5, 0.0, 1.0)", "square": "jnp.square(y)", "abs": "jnp.abs(y)", "negate": "-y", "scale": "y * {c}",
    "shift": "y + {c}", "clamp": "jnp.clip(y, -{b}, {d})", "soft_sign": "y / (1.0 + jnp.abs(y))", "gaussian": "jnp.exp(-0.5 * jnp.square(y))",
}
REDUCE_KEEP = {"softmax": "jax.nn.softmax(y, axis=-1)", "log_softmax": "jax.nn.log_softmax(y, axis=-1)",
               "normalize": "y / (jnp.sqrt(jnp.sum(jnp.square(y), axis=-1, keepdims=True)) + 1e-6)",
               "center": "y - jnp.mean(y, axis=-1, keepdims=True)", "max_keep": "jnp.max(y, axis=-1, keepdims=True)",
               "min_keep": "jnp.min(y, axis=-1, keepdims=True)", "sum_keep": "jnp.sum(y, axis=-1, keepdims=True)",
               "mean_keep": "jnp.mean(y, axis=-1, keepdims=True)",
               "logsumexp_keep": "jax.scipy.special.logsumexp(y, axis=-1, keepdims=True)"}
REDUCE = {"sum": "jnp.sum(y, axis={ax})", "mean": "jnp.mean(y, axis={ax})", "max": "jnp.max(y, axis={ax})", "min": "jnp.min(y, axis={ax})",
          "l2": "jnp.sqrt(jnp.sum(jnp.square(y), axis={ax}))", "var": "jnp.var(y, axis={ax})", "std": "jnp.std(y, axis={ax})",
          "logsumexp": "jax.scipy.special.logsumexp(y, axis={ax})"}
BINARY = {"add": "x + z", "subtract": "x - z", "multiply": "x * z", "maximum": "jnp.maximum(x, z)", "minimum": "jnp.minimum(x, z)",
          "divide": "x / (jnp.abs(z) + 1.0)", "mean_of": "0.5 * (x + z)"}
TILES = (32, 64, 128, 256)


def fill(expr: str, rng) -> str:
    return expr.format(a=round(rng.uniform(0.01, 0.3), 3), b=round(rng.uniform(0.5, 3.0), 2), c=round(rng.uniform(0.2, 2.5), 3), d=round(rng.uniform(0.5, 3.0), 2))


def epilogue(rng, lo=1, hi=4) -> list:
    return [(n, fill(ELEMENTWISE[n], rng)) for n in rng.sample(sorted(ELEMENTWISE), rng.randint(lo, hi))]


def apply_lines(chain: list, indent: str) -> str:
    return "".join(f"{indent}y = {e}\n" for _, e in chain)


def config(d: dict) -> str:
    return "CONFIG = {\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in d.items()) + "}\n"


def dt(rng) -> str:
    return "jnp.bfloat16" if rng.random() < 0.3 else "jnp.float32"


def ref_module(doc: str, cfg: dict, dtype: str, inputs: list, sig: str, body: str) -> str:
    """`inputs`: (name, shape expression, scale). Each array comes from its own split of one key, so the three seeds really differ."""
    keys = ", ".join(f"k{i}" for i in range(len(inputs))) + ("," if len(inputs) == 1 else "")
    draws = "".join(f"    {n} = jax.random.normal(k{i}, {shape}, dtype) * {sc}\n" for i, (n, shape, sc) in enumerate(inputs))
    ret = ", ".join(n for n, _, _ in inputs) + ("," if len(inputs) == 1 else "")
    return (f'"""{doc}"""\nimport jax\nimport jax.numpy as jnp\nfrom jax import lax\n\n{config(cfg)}\n\ndef create_inputs(dtype={dtype}):\n    c = CONFIG\n'
            f"    {keys} = jax.random.split(jax.random.key(0), {len(inputs)})\n{draws}    return {ret}\n\n\ndef workload({sig}):\n{body}")


def _tile(n: int, rng, last: bool = False) -> int:
    """A block size for an axis of length n. The TPU lowering needs the last block dimension divisible by 128 (the one before it by 8) unless it spans the axis."""
    opts = [t for t in TILES if n % t == 0 and n // t <= 4 and (not last or t % 128 == 0)]
    return rng.choice(opts) if opts else n


def _chain_names(chain: list) -> str:
    return " -> ".join(n for n, _ in chain)


def matmul_epilogue(i: int, rng) -> tuple:
    """A (m, k) @ B (k, n), or the batched version with a leading batch axis, then an elementwise chain, tiled over (m, n, k)."""
    batched = rng.random() < 0.3
    M, N, K = (rng.choice((64, 128, 256, 512)) for _ in range(3))
    bn = rng.choice((2, 4, 8)) if batched else 0
    tm, tn, tk = _tile(M, rng), _tile(N, rng, last=True), _tile(K, rng, last=True)
    bias = rng.random() < 0.3
    chain, dtype = epilogue(rng), dt(rng)
    cfg = {"name": f"gen_matmul_{i}", **({"batch": bn} if batched else {}), "m": M, "n": N, "k": K}
    pre = ("c['batch'], " if batched else "")
    inputs = [("a", f"({pre}c['m'], c['k'])", 0.1), ("b", f"({pre}c['k'], c['n'])", 0.1)] + ([("v", "(c['n'],)", 0.1)] if bias else [])
    sig = "a, b" + (", v" if bias else "")
    body = "    y = jnp.matmul(a, b)\n" + ("    y = y + v\n" if bias else "") + apply_lines(chain, "    ") + "    return y\n"
    ref = ref_module(f"{'Batched matmul' if batched else 'Matmul'}{' -> bias' if bias else ''} -> {_chain_names(chain)}.", cfg, dtype, inputs, sig, body)
    # batch axis first in the grid; k last, so the accumulator resets once per output tile
    g = (["b_i"] if batched else []) + ["i", "j", "l"]
    args = ", ".join(g)
    bdim = "None, " if batched else ""
    bix = "b_i, " if batched else ""
    kn = len(g) - 1
    kern = (f"\n\ndef _kernel(a_ref, b_ref{', v_ref' if bias else ''}, o_ref, acc):\n    @pl.when(pl.program_id({kn}) == 0)\n    def _():\n        acc[...] = jnp.zeros_like(acc)\n"
            "    acc[...] += jnp.dot(a_ref[...].astype(jnp.float32), b_ref[...].astype(jnp.float32), preferred_element_type=jnp.float32)\n\n"
            f"    @pl.when(pl.program_id({kn}) == pl.num_programs({kn}) - 1)\n    def _():\n        y = acc[...]\n" + ("        y = y + v_ref[...].astype(jnp.float32)\n" if bias else "")
            + apply_lines(chain, "        ") + "        o_ref[...] = y.astype(o_ref.dtype)\n")
    shape_b = "B, " if batched else ""
    head = ("    B, M, K = a.shape\n    _, N = b.shape[1:]\n" if batched else "    M, K = a.shape\n    _, N = b.shape\n")
    grid = (f"(B, M // {tm}, N // {tn}, K // {tk})" if batched else f"(M // {tm}, N // {tn}, K // {tk})")
    sem = ", ".join(['"parallel"'] * (len(g) - 1) + ['"arbitrary"'])
    specs = (f"pl.BlockSpec(({bdim}{tm}, {tk}), lambda {args}: ({bix}i, l))", f"pl.BlockSpec(({bdim}{tk}, {tn}), lambda {args}: ({bix}l, j))")
    vspec = f", pl.BlockSpec(({tn},), lambda {args}: (j,))" if bias else ""
    code = (f"{HEADER_K}{kern}\n\ndef workload({sig}):\n{head}    return pl.pallas_call(\n        _kernel,\n        out_shape=jax.ShapeDtypeStruct(({shape_b}M, N), a.dtype),\n"
            f"        grid={grid},\n        in_specs=[{specs[0]}, {specs[1]}{vspec}],\n        out_specs=pl.BlockSpec(({bdim}{tm}, {tn}), lambda {args}: ({bix}i, j)),\n"
            f"        scratch_shapes=[pltpu.VMEM(({tm}, {tn}), jnp.float32)],\n        compiler_params=pltpu.CompilerParams(dimension_semantics=({sem})),\n    )({sig})\n")
    return ("bmm_epilogue" if batched else "matmul_epilogue"), ref, code


def row_reduction(i: int, rng) -> tuple:
    """A row reduction over the last axis, of a plain array or of a matmul: whole rows in each block, tiled over rows only."""
    with_matmul = rng.random() < 0.55
    M, N = rng.choice((64, 128, 256, 512)), rng.choice((64, 128, 256))
    K = rng.choice((64, 128, 256))
    tm, tk = _tile(M, rng), _tile(K, rng, last=True)
    pre, post = epilogue(rng, 0, 2), epilogue(rng, 0, 1)
    rname = rng.choice(sorted(REDUCE_KEEP))
    keeps = REDUCE_KEEP[rname].startswith("jax.nn") or rname in ("normalize", "center")
    dtype = dt(rng)
    Nout = "N" if keeps else "1"
    chain = pre + [(rname, REDUCE_KEEP[rname])] + post
    cfg = {"name": f"gen_rowreduce_{i}", "m": M, "n": N, **({"k": K} if with_matmul else {})}
    if with_matmul:
        inputs, sig, first = [("a", "(c['m'], c['k'])", 0.1), ("b", "(c['k'], c['n'])", 0.1)], "a, b", "    y = jnp.matmul(a, b)\n"
    else:
        inputs, sig, first = [("x", "(c['m'], c['n'])", 1.0)], "x", "    y = x\n"
    ref = ref_module(f"{'Matmul -> ' if with_matmul else ''}{_chain_names(chain)} over the last axis.", cfg, dtype, inputs, sig, first + apply_lines(chain, "    ") + "    return y\n")
    if with_matmul:
        kern = ("\n\ndef _kernel(a_ref, b_ref, o_ref, acc):\n    @pl.when(pl.program_id(1) == 0)\n    def _():\n        acc[...] = jnp.zeros_like(acc)\n"
                "    acc[...] += jnp.dot(a_ref[...].astype(jnp.float32), b_ref[...].astype(jnp.float32), preferred_element_type=jnp.float32)\n\n"
                "    @pl.when(pl.program_id(1) == pl.num_programs(1) - 1)\n    def _():\n        y = acc[...]\n" + apply_lines(chain, "        ") + "        o_ref[...] = y.astype(o_ref.dtype)\n")
        tail = (f"    return pl.pallas_call(\n        _kernel,\n        out_shape=jax.ShapeDtypeStruct((M, {Nout}), a.dtype),\n        grid=(M // {tm}, K // {tk}),\n"
                f"        in_specs=[pl.BlockSpec(({tm}, {tk}), lambda i, l: (i, l)), pl.BlockSpec(({tk}, N), lambda i, l: (l, 0))],\n"
                f"        out_specs=pl.BlockSpec(({tm}, {Nout}), lambda i, l: (i, 0)),\n        scratch_shapes=[pltpu.VMEM(({tm}, N), jnp.float32)],\n"
                '        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),\n    )(a, b)\n')
        code = f"{HEADER_K}{kern}\n\ndef workload(a, b):\n    M, K = a.shape\n    _, N = b.shape\n{tail}"
    else:
        kern = "\n\ndef _kernel(x_ref, o_ref):\n    y = x_ref[...].astype(jnp.float32)\n" + apply_lines(chain, "    ") + "    o_ref[...] = y.astype(o_ref.dtype)\n"
        tail = (f"    return pl.pallas_call(\n        _kernel,\n        out_shape=jax.ShapeDtypeStruct((M, {Nout}), x.dtype),\n        grid=(M // {tm},),\n"
                f"        in_specs=[pl.BlockSpec(({tm}, N), lambda i: (i, 0))],\n        out_specs=pl.BlockSpec(({tm}, {Nout}), lambda i: (i, 0)),\n"
                '        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),\n    )(x)\n')
        code = f"{HEADER_K}{kern}\n\ndef workload(x):\n    M, N = x.shape\n{tail}"
    return "row_reduction", ref, code


def _dims(rank: int, rng) -> list:
    return [rng.choice((16, 32, 64, 128)) for _ in range(rank)]


def _capped(dims: list, limit: int = 1 << 23) -> list:
    """Halve the largest non-leading axis until the array has at most `limit` elements: the checks hold it in memory on a 17 GB Mac."""
    while True:
        n = 1
        for d in dims:
            n *= d
        if n <= limit:
            return dims
        k = max(range(1, len(dims)), key=lambda i: dims[i])
        dims[k] //= 2


def elementwise(i: int, rng) -> tuple:
    """A unary chain or a binary op then a chain, on a rank 2 to 4 array, tiled over the first axis."""
    rank = rng.randint(2, 4)
    dims = _dims(rank, rng)
    dims[0] = rng.choice((64, 128, 256, 512))
    dims = _capped(dims)
    t = _tile(dims[0], rng)
    binary = rng.random() < 0.6
    chain, dtype = epilogue(rng, 1, 4), dt(rng)
    bname = rng.choice(sorted(BINARY))
    names = [f"d{k}" for k in range(rank)]
    shape = "(" + ", ".join(f"c[{n!r}]" for n in names) + ",)"
    cfg = {"name": f"gen_elementwise_{i}", **dict(zip(names, dims))}
    inputs = [("x", shape, 1.0)] + ([("z", shape, 1.0)] if binary else [])
    sig = "x, z" if binary else "x"
    first = f"    y = {BINARY[bname]}\n" if binary else "    y = x\n"
    ref = ref_module(f"{bname + ' -> ' if binary else ''}{_chain_names(chain)} on a rank {rank} array.", cfg, dtype, inputs, sig, first + apply_lines(chain, "    ") + "    return y\n")
    block = f"({t}, " + ", ".join(str(d) for d in dims[1:]) + ")"
    ix = "(i, " + ", ".join("0" for _ in dims[1:]) + ")"
    kfirst = ("    x = x_ref[...].astype(jnp.float32)\n    z = z_ref[...].astype(jnp.float32)\n" + f"    y = {BINARY[bname]}\n") if binary else "    y = x_ref[...].astype(jnp.float32)\n"
    kern = f"\n\ndef _kernel(x_ref, {'z_ref, ' if binary else ''}o_ref):\n{kfirst}{apply_lines(chain, '    ')}    o_ref[...] = y.astype(o_ref.dtype)\n"
    spec = f"pl.BlockSpec({block}, lambda i: {ix})"
    code = (f"{HEADER_K}{kern}\n\ndef workload({sig}):\n    return pl.pallas_call(\n        _kernel,\n        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),\n        grid=(x.shape[0] // {t},),\n"
            f"        in_specs=[{spec}{', ' + spec if binary else ''}],\n        out_specs={spec},\n"
            f'        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),\n    )({sig})\n')
    return "elementwise", ref, code


def reduction(i: int, rng) -> tuple:
    """A reduction over some non-leading axes of a rank 2 to 4 array, tiled over the first axis."""
    rank = rng.randint(2, 4)
    dims = _dims(rank, rng)
    dims[0] = rng.choice((64, 128, 256, 512))
    dims = _capped(dims)
    axes = sorted(rng.sample(range(1, rank), rng.randint(1, rank - 1)))
    t = _tile(dims[0], rng, last=len(axes) == rank - 1)
    rname, chain, dtype = rng.choice(sorted(REDUCE)), epilogue(rng, 0, 2), dt(rng)
    post = epilogue(rng, 0, 1)
    names = [f"d{k}" for k in range(rank)]
    shape = "(" + ", ".join(f"c[{n!r}]" for n in names) + ",)"
    cfg = {"name": f"gen_reduction_{i}", **dict(zip(names, dims))}
    ax = "(" + ", ".join(map(str, axes)) + ",)"
    rx = REDUCE[rname].format(ax=ax)
    out = [d for k, d in enumerate(dims) if k not in axes]
    ref = ref_module(f"{_chain_names(chain + [(rname, rx)] + post)} over axes {ax}.", cfg, dtype, [("x", shape, 1.0)], "x",
                     "    y = x\n" + apply_lines(chain, "    ") + f"    y = {rx}\n" + apply_lines(post, "    ") + "    return y\n")
    kern = (f"\n\ndef _kernel(x_ref, o_ref):\n    y = x_ref[...].astype(jnp.float32)\n{apply_lines(chain, '    ')}    y = {rx}\n{apply_lines(post, '    ')}    o_ref[...] = y.astype(o_ref.dtype)\n")
    block_in = f"({t}, " + ", ".join(map(str, dims[1:])) + ")"
    ix_in = "(i, " + ", ".join("0" for _ in dims[1:]) + ")"
    block_out = "(" + ", ".join(map(str, [t] + out[1:])) + ("," if len(out) == 1 else "") + ")"
    ix_out = "(" + ", ".join(["i"] + ["0"] * (len(out) - 1)) + ("," if len(out) == 1 else "") + ")"
    code = (f"{HEADER_K}{kern}\n\ndef workload(x):\n    return pl.pallas_call(\n        _kernel,\n        out_shape=jax.ShapeDtypeStruct({tuple(out)!r}, x.dtype),\n        grid=(x.shape[0] // {t},),\n"
            f"        in_specs=[pl.BlockSpec({block_in}, lambda i: {ix_in})],\n        out_specs=pl.BlockSpec({block_out}, lambda i: {ix_out}),\n"
            f'        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),\n    )(x)\n')
    return "reduction", ref, code


def conv_im2col(i: int, rng) -> tuple:
    """A VALID 1D, 2D or 3D convolution, batched over the first axis: im2col by slicing inside the kernel, then a matmul with the flattened weights."""
    nd = rng.choice((1, 2, 2, 2, 3))
    B, Ci, Co = rng.choice((2, 4, 8, 16)), rng.choice((1, 3, 4, 8)), rng.choice((4, 8, 16, 32))
    ks = [rng.choice((1, 3, 5) if nd < 3 else (1, 3)) for _ in range(nd)]
    sp = [rng.choice((16, 24, 32, 48)) if nd < 3 else rng.choice((8, 12, 16)) for _ in range(nd)]
    chain, dtype, bias = epilogue(rng, 0, 3), dt(rng), rng.random() < 0.4
    letters = "HWD"[:nd]
    spatial = "".join(letters)
    cfg = {"name": f"gen_conv{nd}d_{i}", "batch": B, "in_ch": Ci, "out_ch": Co, "spatial": tuple(sp), "kernel": tuple(ks)}
    inputs = [("x", "(c['batch'], c['in_ch']) + c['spatial']", 0.5), ("w", "c['kernel'] + (c['in_ch'], c['out_ch'])", 0.05)] + ([("v", "(c['out_ch'],)", 0.05)] if bias else [])
    sig = "x, w" + (", v" if bias else "")
    vshape = "(-1, " + ", ".join("1" for _ in range(nd)) + ")"
    body = (f"    y = lax.conv_general_dilated(x, w, (1,) * {nd}, 'VALID', dimension_numbers=('NC{spatial}', '{spatial}IO', 'NC{spatial}'))\n"
            + (f"    y = y + v.reshape{vshape}\n" if bias else "") + apply_lines(chain, "    ") + "    return y\n")
    ref = ref_module(f"Conv{nd}d{' -> bias' if bias else ''}{' -> ' + _chain_names(chain) if chain else ''}.", cfg, dtype, inputs, sig, body)
    outsp = [s - k + 1 for s, k in zip(sp, ks)]
    P = 1
    for o in outsp:
        P *= o
    nk = 1
    for k in ks:
        nk *= k
    slices = ", ".join(f"d{n}:d{n} + {outsp[n]}" for n in range(nd))
    loops = "".join(f"{'    ' * (n + 1)}for d{n} in range({ks[n]}):\n" for n in range(nd))
    kern = ("\n\ndef _kernel(x_ref, w_ref" + (", v_ref" if bias else "") + ", o_ref):\n    x = x_ref[...].astype(jnp.float32)\n    cols = []\n" + loops
            + f"{'    ' * (nd + 1)}cols.append(x[:, {slices}].reshape({Ci}, {P}))\n"
            + f"    patches = jnp.concatenate(cols, axis=0)\n    weights = w_ref[...].astype(jnp.float32).reshape({nk * Ci}, {Co})\n"
            + "    y = jnp.dot(patches.T, weights, preferred_element_type=jnp.float32)\n"
            + ("    y = y + v_ref[...].astype(jnp.float32)\n" if bias else "")
            + f"    y = y.T.reshape(({Co},) + {tuple(outsp)!r})\n"
            + apply_lines(chain, "    ") + "    o_ref[...] = y.astype(o_ref.dtype)\n")
    # VALID windows: the block keeps the batch axis out and reads the whole sample
    x_block = "(None, " + ", ".join(map(str, [Ci] + sp)) + ")"
    x_ix = "(b, " + ", ".join("0" for _ in range(nd + 1)) + ")"
    w_shape = tuple(ks) + (Ci, Co)
    w_ix = "(" + ", ".join("0" for _ in w_shape) + ")"
    o_block = "(None, " + ", ".join(map(str, [Co] + outsp)) + ")"
    in_specs = f"pl.BlockSpec({x_block}, lambda b: {x_ix}), pl.BlockSpec({w_shape!r}, lambda b: {w_ix})" + (f", pl.BlockSpec(({Co},), lambda b: (0,))" if bias else "")
    code = (f"{HEADER_K}{kern}\n\ndef workload({sig}):\n    return pl.pallas_call(\n        _kernel,\n        out_shape=jax.ShapeDtypeStruct((x.shape[0], {Co}) + {tuple(outsp)!r}, x.dtype),\n        grid=(x.shape[0],),\n"
            f"        in_specs=[{in_specs}],\n        out_specs=pl.BlockSpec({o_block}, lambda b: {x_ix}),\n"
            f'        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),\n    )({sig})\n')
    return f"conv{nd}d_im2col", ref, code


# share of candidates per family; the build step takes the survivors down to the target count
FAMILIES = [(matmul_epilogue, 0.36), (row_reduction, 0.14), (elementwise, 0.16), (reduction, 0.16), (conv_im2col, 0.18)]


def candidates(n: int, seed: int) -> list:
    out = []
    for f, share in FAMILIES:
        for k in range(round(n * share)):
            rng = random.Random(f"{seed}/{f.__name__}/{k}")
            fam, ref, code = f(k, rng)
            cid = f"gen__{f.__name__}__{k:05d}"
            out.append({"id": cid, "task": cid, "family": fam, "ref_code": ref, "code": code})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=2400); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-per-family", type=int, default=0, help="a small sample: the first K of each generator, written to generated_sample*")
    ap.add_argument("--python", default=None)
    a = ap.parse_args(argv)
    cands = candidates(a.n, a.seed)
    if a.limit_per_family:
        seen, keep = collections.Counter(), []
        for c in cands:
            key = c["id"].split("__")[1]
            seen[key] += 1
            if seen[key] <= a.limit_per_family:
                keep.append(c)
        cands = keep
    stem = "generated_sample" if a.limit_per_family else "generated"
    write_jsonl(V3 / f"{stem}_candidates.jsonl", cands)
    res = verify(cands, V3 / f"{stem}_verified.jsonl", a.python, log=lambda m: print(m, flush=True))
    by = collections.Counter((c["family"], res[c["id"]].get("stage") if not res[c["id"]].get("ok") else "OK") for c in cands)
    for k, v in sorted(by.items()):
        print(k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
