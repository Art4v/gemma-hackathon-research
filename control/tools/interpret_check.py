"""Interpret-mode numerics check for a near-miss kernel against a shrunken copy of ref.py."""
import functools, pathlib, types
import numpy as np
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl

jax.config.update("jax_default_matmul_precision", "highest")
_orig = pl.pallas_call
pl.pallas_call = functools.partial(_orig, interpret=True)


def _load(path, name, subs):
    src = pathlib.Path(path).read_text()
    for a, b in subs:
        assert a in src, f"substitution target not found in {path}: {a!r}"
        src = src.replace(a, b)
    m = types.ModuleType(name)
    m.__file__ = str(path)
    exec(compile(src, str(path), "exec"), m.__dict__)
    return m


def run(d, ref_subs=(), ker_subs=(), make_inputs=None):
    d = pathlib.Path(d)
    ref = _load(d / "ref.py", "ref_small", ref_subs)
    ker = _load(d / "kernel.py", "ker_small", ker_subs)
    inputs = make_inputs(ref) if make_inputs else ref.create_inputs()
    nbytes = sum(np.asarray(a).nbytes for a in jax.tree_util.tree_leaves(inputs))
    print(f"input MB: {nbytes / 2**20:.1f}")
    f32 = jax.tree_util.tree_map(lambda a: a.astype(jnp.float32) if jnp.issubdtype(a.dtype, jnp.floating) else a, inputs)
    want = ref.workload(*f32)
    got = ker.workload(*inputs)
    wl, gl = jax.tree_util.tree_leaves(want), jax.tree_util.tree_leaves(got)
    assert len(wl) == len(gl), f"output count {len(gl)} != ref {len(wl)}"
    ok = True
    for w, g in zip(wl, gl):
        w, g = np.asarray(w, np.float32), np.asarray(g, np.float32)
        if w.shape != g.shape:
            print(f"SHAPE MISMATCH got {g.shape} want {w.shape}"); ok = False; continue
        err = np.abs(w - g)
        close = np.allclose(g, w, rtol=1e-2, atol=1e-2)
        print(f"shape {w.shape} max_abs_err {err.max():.3g} max|ref| {np.abs(w).max():.3g} allclose={close}")
        ok &= bool(close)
    print("MATCH" if ok else "MISMATCH")
    return ok
