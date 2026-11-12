"""Batch op-chain tracer, same walk as aarav/dataset-v4 dataset/scripts/signature.py (oracle outside_ops ALLOWED + _CANON, collapse):
python -I trace_child.py ORACLE_DIR IN.jsonl OUT.jsonl  (IN rows {key, ref}); abstract inputs only (jax.eval_shape), 60 s per ref."""
import ast, json, re, signal, sys, types
sys.path.insert(0, sys.argv[1])
import jax, jax.numpy as jnp  # noqa: F401
from outside_ops import ALLOWED, _sub_jaxprs
_CANON = {"pjit": None, "jit": None, "closed_call": None, "custom_jvp_call": None, "custom_vjp_call": None, "custom_vjp_call_jaxpr": None,
          "remat": None, "checkpoint": None, "integer_pow": "pow", "logistic": "sigmoid"}
def canonical(name):
    n = name.lower()
    if n.endswith("_p"): n = n[:-2]
    if n in ALLOWED: return None
    return _CANON.get(n, n) if n in _CANON else n
def walk_ops(jaxpr, out=None):
    out = [] if out is None else out
    for e in jaxpr.eqns:
        name = e.primitive.name
        if name == "pallas_call": out.append("pallas_call"); continue
        subs = list(_sub_jaxprs(e.params))
        if subs:
            for s in subs: walk_ops(s, out)
            continue
        c = canonical(name)
        if c is not None: out.append(c)
    return out
def collapse(ops):
    out = []
    for o in ops:
        if not out or out[-1] != o: out.append(o)
    return out
_STATIC = re.compile(r"jax\.jit\(\s*workload\s*,\s*static_argnums\s*=\s*(\([^)]*\)|\[[^\]]*\]|\d+)")
def alarm(*_): raise TimeoutError("trace timeout")
signal.signal(signal.SIGALRM, alarm)
with open(sys.argv[3], "w") as fo:
    for line in open(sys.argv[2]):
        x = json.loads(line); src = x["ref"]
        try:
            signal.alarm(60)
            mod = types.ModuleType("ref"); exec(compile(src, "ref.py", "exec"), mod.__dict__)
            m = _STATIC.search(src); static = ()
            if m:
                v = ast.literal_eval(m.group(1)); static = (v,) if isinstance(v, int) else tuple(v)
            held = {}
            def build():
                xs = mod.create_inputs(); held.update((i, xs[i]) for i in static if i < len(xs)); return xs
            abstract = jax.eval_shape(build)
            if held: abstract = type(abstract)(held.get(i, a) for i, a in enumerate(abstract))
            jaxpr = jax.make_jaxpr(mod.workload, static_argnums=tuple(sorted(held)))(*abstract).jaxpr
            fo.write(json.dumps({"key": x["key"], "op_chain": collapse(walk_ops(jaxpr))}) + "\n")
        except BaseException as e:
            fo.write(json.dumps({"key": x["key"], "error": f"{type(e).__name__}: {str(e)[:200]}"}) + "\n")
        finally:
            signal.alarm(0)
        fo.flush()
