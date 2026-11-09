"""Pallas API coverage of a kernel dataset: which structural features each verified kernel uses, how many distinct skeletons there
are, and how concentrated the rows are. Static (ast + regex), runs nothing.
  python3 coverage.py ROWS.jsonl [ROWS.jsonl ...]     (dataset-v1 rows: turns[-1].code; v3 rows: messages[-1].content)
"""
import ast, collections, json, re, sys

FLAGS = {  # feature: regex on the kernel source
    "accumulator_scratch": r"scratch_shapes",
    "arbitrary_axis": r"[\"']arbitrary[\"']",
    "pl_when": r"pl\.when\(",
    "scalar_prefetch": r"PrefetchScalarGridSpec|num_scalar_prefetch",
    "manual_dma": r"make_async_copy|async_copy|memory_space\s*=\s*(pl|pltpu)\.ANY",
    "smem": r"SMEM",
    "masking_iota": r"iota\(",
    "cdiv_grid": r"pl\.cdiv|cdiv\(",
    "in_kernel_loop": r"fori_loop|lax\.scan|while_loop",
    "dynamic_slice": r"pl\.ds\(|pl\.dslice\(",
    "aliasing": r"input_output_aliases",
    "mxu_dot": r"jnp\.dot|lax\.dot_general|jnp\.matmul|pl\.dot|jnp\.einsum",
    "f32_accumulate": r"preferred_element_type",
    "vmem_limit": r"vmem_limit_bytes",
    "emit_pipeline": r"emit_pipeline",
    "bf16": r"bfloat16",
}


def code_of(row):
    if "turns" in row:
        return row["turns"][-1]["code"]
    m = re.findall(r"```(?:python)?\n(.*?)```", row["messages"][-1]["content"], re.S)
    return m[-1] if m else ""


def tuple_len(node, assigns):
    if isinstance(node, ast.Name) and node.id in assigns:
        node = assigns[node.id]
    return len(node.elts) if isinstance(node, (ast.Tuple, ast.List)) else None


def features(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    assigns = {t.id: n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    f = {"pallas_calls": src.count("pallas_call("), "grid_rank": set(), "index_map_arity": set(), "n_in_specs": set(), "n_outputs": set()}
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        name = ast.unparse(n.func).split(".")[-1]
        kw = {k.arg: k.value for k in n.keywords}
        if "grid" in kw:
            f["grid_rank"].add(tuple_len(kw["grid"], assigns))
        if "in_specs" in kw:
            f["n_in_specs"].add(tuple_len(kw["in_specs"], assigns))
        if name == "pallas_call" and "out_shape" in kw:
            f["n_outputs"].add(tuple_len(kw["out_shape"], assigns) or 1)
        if name == "BlockSpec":
            for a in list(n.args) + list(kw.values()):
                if isinstance(a, ast.Lambda):
                    f["index_map_arity"].add(len(a.args.args))
    for k, rx in FLAGS.items():
        f[k] = bool(re.search(rx, src))
    return f


def skeleton(f):
    """Structure only: grid ranks, index-map arities, outputs, number of pallas_calls, and the flags (ops, shapes, names ignored)."""
    s = lambda v: tuple(sorted(x for x in v if x is not None))
    return (s(f["grid_rank"]), s(f["index_map_arity"]), s(f["n_outputs"]), min(f["pallas_calls"], 3),
            tuple(k for k in FLAGS if f[k] and k not in ("bf16", "vmem_limit")))


def report(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if r.get("kind", "single") in ("single", "generated", "shape_variant", "correct_slow", "cpu")]
    fs = [f for f in (features(code_of(r)) for r in rows) if f]
    n = len(fs)
    print(f"\n== {path}: {n} kernels")
    cnt = lambda key: collections.Counter(x for f in fs for x in f[key] if x is not None)
    for key in ("grid_rank", "index_map_arity", "n_outputs"):
        print(f"  {key:16s} " + "  ".join(f"{k}:{v}" for k, v in sorted(cnt(key).items())))
    print("  multi pallas_call  " + str(sum(f["pallas_calls"] > 1 for f in fs)))
    for k in FLAGS:
        c = sum(f[k] for f in fs)
        print(f"  {k:20s} {c:4d}  {100 * c / max(1, n):5.1f}%")
    sk = collections.Counter(skeleton(f) for f in fs)
    top10 = sum(c for _, c in sk.most_common(10))
    print(f"  distinct skeletons {len(sk)}; top-10 skeletons hold {top10} rows ({100 * top10 / max(1, n):.0f}%)")


if __name__ == "__main__":
    assert skeleton(features("pl.pallas_call(k, grid=(2, 3), in_specs=[pl.BlockSpec((8, 128), lambda i, j: (i, j))], "
                             "out_shape=jax.ShapeDtypeStruct((8, 8), jnp.float32))(x)"))[:3] == ((2,), (2,), (1,))
    for p in sys.argv[1:]:
        report(p)
