"""Op-chain signatures of baseline modules, for the contamination filter (PROCESS.md section 3, layer 2).

  python dataset/scripts/signature.py exam [--refresh]     signatures of the 50 exam references, cached in problems/exam_signatures.json
  python dataset/scripts/signature.py one FILE.py          the signature of one module

A signature is computed by tracing `workload` on abstract inputs in a `python -I` child (common.Isolated) on the CPU JAX platform;
the module is never imported in-process. The walk uses the oracle's own outside_ops.py (`_sub_jaxprs`, `ALLOWED`) so what counts as
layout is exactly what the oracle lets through. The pure parts (canonicalise, collapse, walk on a fake jaxpr) import without JAX.
"""
from __future__ import annotations

import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import BENCH, Isolated, ORACLE, PROBLEMS, exam_files, read_json, write_json  # noqa: E402

EXAM_SIG_CACHE = PROBLEMS / "exam_signatures.json"
TIMEOUT_S = 180

# primitive name -> canonical name; names not here keep their own (lower-cased, trailing _p stripped). None = skipped.
_CANON = {"pjit": None, "jit": None, "closed_call": None, "custom_jvp_call": None, "custom_vjp_call": None, "custom_vjp_call_jaxpr": None,
          "remat": None, "checkpoint": None, "integer_pow": "pow", "logistic": "sigmoid"}
# The ALLOWED set is read from the oracle at import (pure: outside_ops imports without JAX)
sys.path.insert(0, str(ORACLE))
from outside_ops import ALLOWED, _sub_jaxprs  # noqa: E402


def canonical(name: str) -> str | None:
    n = name.lower()
    if n.endswith("_p"):
        n = n[:-2]
    if n in ALLOWED:
        return None
    return _CANON.get(n, n) if n in _CANON else n


def walk_ops(jaxpr, out: list | None = None) -> list:
    """Canonical primitive names outside pallas_call, in program order, descending through sub-jaxprs (never into a kernel body)."""
    out = [] if out is None else out
    for e in jaxpr.eqns:
        name = e.primitive.name
        if name == "pallas_call":
            out.append("pallas_call")
            continue
        subs = list(_sub_jaxprs(e.params))
        if subs:
            for s in subs:
                walk_ops(s, out)
            continue
        c = canonical(name)
        if c is not None:
            out.append(c)
    return out


def collapse(ops: list) -> list:
    out = []
    for o in ops:
        if not out or out[-1] != o:
            out.append(o)
    return out


def multiset(ops: list) -> dict:
    m: dict = {}
    for o in ops:
        m[o] = m.get(o, 0) + 1
    return dict(sorted(m.items()))


def jaccard(a: dict, b: dict) -> float:
    """Weighted Jaccard of two multisets."""
    keys = set(a) | set(b)
    inter = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return inter / union if union else 0.0


CHILD = r'''
import ast, json, re, sys, types
import jax, jax.numpy as jnp, numpy as np
sys.path.insert(0, ".")
import outside_ops
import sigpure

src = open("ref.py").read()
mod = types.ModuleType("ref"); sys.modules["ref"] = mod
exec(compile(src, "ref.py", "exec"), mod.__dict__)

_STATIC = re.compile(r"jax\.jit\(\s*workload\s*,\s*static_argnums\s*=\s*(\([^)]*\)|\[[^\]]*\]|\d+)")
m = _STATIC.search(src)
static = ()
if m:
    v = ast.literal_eval(m.group(1)); static = (v,) if isinstance(v, int) else tuple(v)
held = {}
def build():
    xs = mod.create_inputs()
    held.update((i, xs[i]) for i in static if i < len(xs))
    return xs
abstract = jax.eval_shape(build)
if held:
    abstract = type(abstract)(held.get(i, x) for i, x in enumerate(abstract))
leaves = [x for x in jax.tree_util.tree_leaves(abstract) if hasattr(x, "shape")]
jaxpr = jax.make_jaxpr(mod.workload, static_argnums=tuple(sorted(held)))(*abstract).jaxpr
ops = sigpure.walk_ops(jaxpr)
out = jax.eval_shape(jax.jit(mod.workload, static_argnums=tuple(sorted(held))), *abstract)
out_leaves = [x for x in jax.tree_util.tree_leaves(out) if hasattr(x, "shape")]
print(json.dumps({"op_chain": sigpure.collapse(ops), "op_multiset": sigpure.multiset(ops),
                  "input_shapes": [list(x.shape) for x in leaves], "input_dtypes": [str(x.dtype) for x in leaves],
                  "output_shapes": [list(x.shape) for x in out_leaves],
                  "input_bytes": int(sum(x.size * x.dtype.itemsize for x in leaves))}))
'''

# the pure walk, shipped to the child as sigpure.py so the child and this module share one definition
_SIGPURE = '''
import sys
from outside_ops import ALLOWED, _sub_jaxprs
_CANON = %r
def canonical(name):
    n = name.lower()
    if n.endswith("_p"): n = n[:-2]
    if n in ALLOWED: return None
    return _CANON.get(n, n) if n in _CANON else n
def walk_ops(jaxpr, out=None):
    out = [] if out is None else out
    for e in jaxpr.eqns:
        name = e.primitive.name
        if name == "pallas_call":
            out.append("pallas_call"); continue
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
def multiset(ops):
    m = {}
    for o in ops: m[o] = m.get(o, 0) + 1
    return dict(sorted(m.items()))
''' % (_CANON,)


def signature_of(ref_code: str, python: str | None = None) -> dict:
    """{op_chain, op_multiset, input_shapes, input_dtypes, output_shapes, input_bytes} or {"error": str}."""
    iso = Isolated(python, timeout=TIMEOUT_S)
    files = {"child.py": CHILD, "ref.py": ref_code, "outside_ops.py": (ORACLE / "outside_ops.py").read_text(), "sigpure.py": _SIGPURE}
    r = iso.run(["child.py"], files=files)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        msg = tail[-1] if tail else f"exit {r.returncode}"
        return {"error": msg[:300]}
    try:
        line = [l for l in r.stdout.splitlines() if l.startswith("{")][-1]
        return json.loads(line)
    except Exception as e:
        return {"error": f"{type(e).__name__}: no JSON from child: {(r.stdout or '')[-200:]}"[:300]}


def exam_signatures(cache_path: pathlib.Path = EXAM_SIG_CACHE, refresh: bool = False, python: str | None = None) -> dict:
    """task -> signature (or {"error"}) for the 50 level1 exam references. Cached."""
    if cache_path.exists() and not refresh:
        return read_json(cache_path)
    out = {}
    for task, level, p in exam_files():
        if level != "level1":
            continue
        out[task] = signature_of(p.read_text(), python)
    write_json(cache_path, out)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("exam"); e.add_argument("--refresh", action="store_true"); e.add_argument("--python", default=None)
    o = sub.add_parser("one"); o.add_argument("file"); o.add_argument("--python", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "one":
        print(json.dumps(signature_of(pathlib.Path(a.file).read_text(), a.python), indent=1))
        return
    sigs = exam_signatures(refresh=a.refresh, python=a.python)
    ok = 0
    for task, s in sorted(sigs.items()):
        if "error" in s:
            print(f"{task:60} ERROR {s['error'][:100]}")
        else:
            ok += 1
            print(f"{task:60} chain {len(s['op_chain']):3}  ops {sum(s['op_multiset'].values()):4}  in {s['input_bytes'] / 1e9:6.3f} GB")
    print(f"{ok} of {len(sigs)} exam references traced")


if __name__ == "__main__":
    main()
