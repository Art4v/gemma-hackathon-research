"""Which manifest tasks have integer, bool or static inputs that never change across the oracle's seeds. CPU only, no chip time.

  python oracle/int_inputs.py [--manifest oracle/manifest.json] [--bench benchmarks/jaxbench] [--seeds 3]

One JSON line per task: int_inputs_vary (after the task's structure transform, exactly as the oracle's rows compute it), n_int_leaves,
constant (each constant leaf: index, dtype, shape, summary), structure_seed, const_float_leaves (the floating leaves seeds >= 1
redraw, per structure.py: leaf, dtype, shape, value); or int_inputs_vary null and unknown with the reason. The oracle also drops a
leaf whose draw makes the reference non-finite; that needs the reference run at full size, so this listing does not check it.
A task's reference is its plain_ref, or for a level 2-only task its human kernel's create_inputs (the level 2 stand-in). Only
create_inputs runs, under jit with the integer leaves as the only outputs, so the float draws are never built; likewise the float
listing computes only the leaves that depend on no PRNG draw (a drawn leaf is not constant).
"""
import argparse, json, pathlib, sys, tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tpu_oracle as o  # noqa: E402

ROOT = o.ROOT


def _draws(jaxpr) -> bool:
    return any(e.primitive.name.startswith(("random_", "threefry")) or any(_draws(j) for j in o._sub_jaxprs(e.params)) for e in jaxpr.eqns)


def const_float_leaves(task: str, ref) -> list:
    """Seed 1's floating leaves (after the task's own transform) that structure.constant_floats picks, found without building the
    drawn ones: a leaf whose value depends on a PRNG draw is skipped, and only the others are computed, on the CPU."""
    f = lambda: o.jax.tree_util.tree_leaves(o.structure.apply(task, o.make_inputs(ref, 1), 1))
    closed, drawn = o.jax.make_jaxpr(f)(), set()
    for e in closed.jaxpr.eqns:
        if e.primitive.name.startswith(("random_", "threefry")) or any(_draws(j) for j in o._sub_jaxprs(e.params)) or \
                any(o._is_var(v) and v in drawn for v in e.invars):
            drawn.update(e.outvars)
    pick = [i for i, (v, a) in enumerate(zip(closed.jaxpr.outvars, closed.out_avals)) if o.jnp.issubdtype(a.dtype, o.jnp.floating)
            and a.size >= 2 and not (o._is_var(v) and v in drawn)]
    with o.jax.default_device(o.jax.devices("cpu")[0]):
        vals = o.jax.jit(lambda: [f()[i] for i in pick])()
    return [{"leaf": i, "dtype": str(x.dtype), "shape": list(x.shape), "value": float(x.reshape(-1)[0])}
            for i, x in zip(pick, vals) if o.structure.constant_floats([x])]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", default=str(ROOT / "manifest.json"))
    ap.add_argument("--bench", default=str(ROOT.parent / "benchmarks/jaxbench"))
    ap.add_argument("--seeds", type=int, default=o.N_SEEDS)
    a = ap.parse_args(argv)
    o.init(cpu_smoke=True)
    seeds, bench = list(range(a.seeds)), pathlib.Path(a.bench)
    for t in json.loads(pathlib.Path(a.manifest).read_text())["tasks"]:
        rel = t.get("plain_ref") or t.get("human_kernel")
        row = {"task": t["task"], "ref": rel, "structure_seed": bool(o.structure.describe(t["task"]))}
        try:
            with tempfile.TemporaryDirectory() as d:
                ref = o.tc.load((bench / rel).read_text(), "ref_" + t["task"].split("_")[0], d)
                vary, const, n = o.int_inputs_vary(t["task"], ref, seeds)
                floats = const_float_leaves(t["task"], ref)
            row.update(int_inputs_vary=vary, n_int_leaves=n, constant=const, const_float_leaves=floats)
        except Exception as e:
            row.update(int_inputs_vary=None, unknown=f"{type(e).__name__}: {str(e)[:300]}")
        print(json.dumps(row), flush=True)
        o.jax.clear_caches()


if __name__ == "__main__":
    main()
