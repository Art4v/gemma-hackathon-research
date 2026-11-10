"""Run in an isolated child: the oracle's step-3 rule on CPU, seed-0 inputs at full size (tpu_oracle.py _key / tpu_check):
  answer key  = the reference on float32-upcast inputs at matmul precision "highest";
  XLA default = the reference on the task's own inputs **as a TPU computes it**: every f32 dot_general / conv operand rounded to bf16
                with f32 accumulation (TPU DEFAULT precision, one MXU pass). A CPU's own default is exact f32, which would make the
                2x-XLA rule far stricter than the chip and push kernels into slow multi-pass matmuls;
  correct if np.allclose(kernel, key, rtol=atol=1e-2), or rel_err(kernel) <= 2 * max(rel_err(XLA default), 1e-6).
The kernel runs in TPU interpret mode (sequential grid: it cannot see ordering bugs; the chip still judges).
  python -I numeric_child.py REF.py KERNEL.py  -> one JSON line"""
import importlib.util, json, sys, time
sys.path.insert(0, "/Users/hrudaynara/Projects/gemma-dataset-claude/oracle")
import numpy as np
import jax, jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu
import tpu_check as tc

MATMULS = ("dot_general", "conv_general_dilated")


def _eval_tpu_default(closed, *args):
    """Evaluate a ClosedJaxpr, rounding f32 matmul/conv operands to bf16 (f32 accumulate); recurse into call-like sub-jaxprs."""
    jaxpr, env = closed.jaxpr, {}
    read = lambda v: v.val if type(v).__name__ == "Literal" else env[v]
    for v, c in zip(jaxpr.constvars, closed.consts): env[v] = c
    for v, a in zip(jaxpr.invars, args): env[v] = a
    for eqn in jaxpr.eqns:
        ins, name, params = [read(v) for v in eqn.invars], eqn.primitive.name, dict(eqn.params)
        if name in MATMULS and any(getattr(x, "dtype", None) == jnp.float32 for x in ins[:2]):
            ins = [x.astype(jnp.bfloat16) if getattr(x, "dtype", None) == jnp.float32 else x for x in ins[:2]] + ins[2:]
            params["preferred_element_type"] = jnp.float32
            if "precision" in params: params["precision"] = None
            outs = eqn.primitive.bind(*ins, **params)
        elif name in ("pjit", "jit", "closed_call", "core_call") and hasattr(params.get("jaxpr"), "jaxpr"):
            outs = _eval_tpu_default(params["jaxpr"], *ins)
        elif name in ("custom_jvp_call", "custom_vjp_call", "custom_vjp_call_jaxpr") and hasattr(params.get("call_jaxpr") or params.get("fun_jaxpr"), "jaxpr"):
            outs = _eval_tpu_default(params.get("call_jaxpr") or params.get("fun_jaxpr"), *ins)
        else:
            outs = eqn.primitive.bind(*ins, **params)
        if not eqn.primitive.multiple_results: outs = [outs]
        for v, o in zip(eqn.outvars, outs): env[v] = o
    return [read(v) for v in jaxpr.outvars]


def tpu_default(fn, inputs):
    closed = jax.make_jaxpr(fn)(*inputs)
    flat = jax.tree.leaves(inputs)
    outs = jax.jit(lambda *a: _eval_tpu_default(closed, *a))(*flat)
    return jax.tree.unflatten(jax.tree.structure(jax.eval_shape(fn, *inputs)), outs)


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


ref, ker = load("ref", sys.argv[1]), load("cand", sys.argv[2])
t0 = time.time()
inputs = ref.create_inputs()
f32 = lambda x: x.astype(jnp.float32) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x
with jax.default_matmul_precision("highest"):
    key = jax.tree.map(np.asarray, jax.jit(ref.workload)(*jax.tree.map(f32, inputs)))
try:
    xla = jax.tree.map(np.asarray, tpu_default(ref.workload, inputs)); xla_mode = "tpu_default_emulated"
except Exception as e:  # fall back to this CPU's own default (stricter than the chip)
    xla = jax.tree.map(np.asarray, jax.jit(ref.workload)(*inputs)); xla_mode = f"cpu_default ({type(e).__name__})"
with pltpu.force_tpu_interpret_mode():
    got = jax.tree.map(np.asarray, ker.workload(*inputs))
kl, gl = jax.tree.leaves(key), jax.tree.leaves(got)
res = {"outputs": len(kl), "xla_mode": xla_mode, "s": None}
if len(kl) != len(gl) or any(np.shape(a) != np.shape(b) for a, b in zip(kl, gl)):
    res.update(ok=False, detail=f"output structure/shapes {[np.shape(b) for b in gl]} != reference {[np.shape(a) for a in kl]}")
else:
    close = tc.close(key, got)
    ek, ex = tc.rel_err(key, got), tc.rel_err(key, xla)
    eq = ek is not None and ex is not None and ek <= 2 * max(ex, 1e-6)
    res.update(ok=bool(close or eq), allclose=bool(close), rel_err_kernel=ek, rel_err_xla_default=ex,
               max_abs_err=max(float(np.abs(np.asarray(b, np.float32) - np.asarray(a, np.float32)).max()) for a, b in zip(kl, gl)))
    if not res["ok"]:
        res["detail"] = f"not allclose (rtol=atol=1e-2) and rel err {ek:.3g} > 2 x XLA default's {ex:.3g} (TPU default precision)"
res["s"] = round(time.time() - t0, 1)
print(json.dumps(res))
