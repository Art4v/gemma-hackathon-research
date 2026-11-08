"""Jaxpr-level outside-kernel check: which JAX primitives run OUTSIDE pallas_call in a candidate's workload?

The AST check (gate.heavy_work_outside_kernel) bans names, so an alias, getattr, `.sum()`, lax.reduce_sum or a product stored in a
variable gets past it (reviewer item 4, issue #1). Traced to a jaxpr, every contraction is `dot_general`, every reduction is `reduce_*`
and every elementwise op is its own primitive, so the rule can be the strict one: outside pallas_call only layout and dtype work.

  python -I outside_ops.py ref.py cand.py      -> RESULT {"outside_ok": bool, "outside": {prim: n}, "outside_disallowed": [...], "pallas_calls": n}
Traces at the checker's abstract input shapes (ref.create_inputs under eval_shape), CPU only, nothing is executed except module import,
so run it in the same sandbox as check.py. The pure parts (walk, classify) are importable without JAX for tests and rescoring.
"""
import json, math, sys

# Layout and dtype primitives: moving or re-labelling data, never computing on it.
ALLOWED = frozenset({"reshape", "transpose", "broadcast_in_dim", "squeeze", "expand_dims", "slice", "dynamic_slice", "dynamic_update_slice",
                     "pad", "concatenate", "convert_element_type", "iota", "copy", "copy_p", "device_put", "rev", "split", "gather",
                     "pjit", "closed_call", "custom_jvp_call", "custom_vjp_call", "custom_vjp_call_jaxpr", "remat", "checkpoint", "jit"})


def _sub_jaxprs(params):
    for v in params.values():
        j = getattr(v, "jaxpr", v)
        if hasattr(j, "eqns"):
            yield j


def walk(jaxpr, outside=None, pallas_calls=0):
    """(primitive counts outside pallas_call, number of pallas_calls). Descends into pjit/cond/scan sub-jaxprs, never into the kernel."""
    outside = {} if outside is None else outside
    for eqn in jaxpr.eqns:
        name = eqn.primitive.name
        if name == "pallas_call":
            pallas_calls += 1
            continue
        subs = list(_sub_jaxprs(eqn.params))
        if subs:
            for s in subs:
                _, pallas_calls = walk(s, outside, pallas_calls)
        else:
            outside[name] = outside.get(name, 0) + 1
    return outside, pallas_calls


def classify(outside, allowed=ALLOWED):
    """(ok, sorted disallowed primitive names)."""
    bad = sorted(p for p in outside if p not in allowed)
    return not bad, bad


# The oracle's rule (oracle/tpu_oracle.py, batch 3). walk, classify and ALLOWED above stay as they are: the frozen study's score() and
# main() use them. R1: gather is no longer free, it moves data by computed indices (a paged-attention page gather done in XLA).
# R2: metadata = small values computed only from non-float inputs. A value is TAINTED if it is a floating (any non-integer, non-bool)
# input of the workload, or an output of any eqn (pallas_call included) with a tainted operand; literals and closed-over constants
# never are. A primitive outside ORACLE_ALLOWED (gather, scatter* and data-addressed dynamic slices too) is allowed only if no operand is tainted and every operand
# and result has at most METADATA_MAX_ELEMENTS elements. Dtype does not matter for untainted values: Megablox's make_group_metadata
# bins int group offsets with jnp.histogram's float32 edges. Float work on a float input stays disallowed however small it is.
# stack and with_memory_space_constraint (Google's 53p and 54p kernels) are layout and placement: no arithmetic on the data.
ORACLE_ALLOWED = (ALLOWED - {"gather"}) | {"stack", "with_memory_space_constraint"}
METADATA_MAX_ELEMENTS = 2 ** 16
_INT_DTYPES = ("int", "uint", "bool")  # str(dtype) prefixes: int4..int64, uint4..uint64, bool
_LOOPS = ("scan", "while")  # a carry feeds back into the body, so one pass of the mapping below is not enough
# where each one's start indices begin among its operands. Shapes are static at trace time, so a start that is not a literal comes
# from the inputs: data-dependent addressing, a gather by another name (Lebron James 6711257577). Those occurrences are counted as
# "<name>(dynamic start)", which is not in ORACLE_ALLOWED, so R2 applies; literal starts stay layout.
_DYNAMIC_STARTS = {"dynamic_slice": 1, "dynamic_update_slice": 2}


def _is_lit(v) -> bool:
    return hasattr(v, "val")  # jaxpr Literals carry .val and are unhashable


def _float_input(v) -> bool:
    """A workload input that taints: anything not integer or bool, or of unknown dtype."""
    aval = getattr(v, "aval", None)
    return aval is None or not str(aval.dtype).startswith(_INT_DTYPES)


def _small(v) -> bool:
    """Literals are scalars by construction and are not sized; a value without an aval is not small."""
    if _is_lit(v):
        return True
    aval = getattr(v, "aval", None)
    return aval is not None and math.prod(aval.shape) <= METADATA_MAX_ELEMENTS


def _walk_taint(jaxpr, tainted_in, outside, heavy) -> tuple:
    """(taint of each outvar, number of pallas_calls), counting into outside and heavy (occurrences that fail R2)."""
    t = {v for v, x in zip(jaxpr.invars, tainted_in) if x and not _is_lit(v)}
    is_t = lambda v: not _is_lit(v) and v in t
    n = 0
    for e in jaxpr.eqns:
        ins, name = [is_t(v) for v in e.invars], e.primitive.name
        outs = [any(ins)] * len(e.outvars)
        subs = [] if name == "pallas_call" else list(_sub_jaxprs(e.params))
        if name == "pallas_call":
            n += 1
        elif len(subs) == 1 and name not in _LOOPS and len(subs[0].invars) == len(e.invars) and len(subs[0].outvars) == len(e.outvars):
            outs, k = _walk_taint(subs[0], ins, outside, heavy)  # pjit, closed_call, custom_jvp/vjp, remat: operand i is invar i
            n += k
        elif subs:
            # ponytail: scan, while and anything else with sub-jaxprs are conservative: every inner input, and every output, is
            # tainted if any operand is. An untainted loop over int metadata still passes.
            for sj in subs:
                n += _walk_taint(sj, [any(ins)] * len(sj.invars), outside, heavy)[1]
        else:
            if name in _DYNAMIC_STARTS and not all(_is_lit(v) for v in e.invars[_DYNAMIC_STARTS[name]:]):
                name = f"{name}(dynamic start)"
            outside[name] = outside.get(name, 0) + 1
            if any(ins) or not all(_small(v) for v in (*e.invars, *e.outvars)):
                heavy[name] = heavy.get(name, 0) + 1
        t.update(v for v, x in zip(e.outvars, outs) if x and not _is_lit(v))
    return [is_t(v) for v in jaxpr.outvars], n


def walk_avals(jaxpr):
    """walk(), plus {primitive: occurrences that fail R2} outside pallas_call: (outside, heavy, pallas_calls). jaxpr is the
    workload's own (its invars are the workload's inputs)."""
    outside, heavy = {}, {}
    _, n = _walk_taint(jaxpr, [_float_input(v) for v in jaxpr.invars], outside, heavy)
    return outside, heavy, n


def classify_avals(outside, heavy, allowed=ORACLE_ALLOWED):
    """(ok, sorted disallowed names, sorted names allowed only as R2 metadata). A primitive outside `allowed` is disallowed if any one
    of its occurrences fails R2."""
    extra = [p for p in outside if p not in allowed]
    bad = sorted(p for p in extra if heavy.get(p))
    return not bad, bad, sorted(p for p in extra if not heavy.get(p))


def _load(path, name):
    import importlib.util
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def shrink_shape(shape, cap=256):
    """check.py slices every array input to at most 256 per dimension; the trace must see the same program the checker ran."""
    return tuple(min(d, cap) for d in shape)


def score(ref_path, cand_path, jax, jnp):
    """The verdict dict for one candidate against one reference, both as files, traced at the checker's shrunk shapes. Any failure
    (import error, trace error, timeout from the caller's alarm) is a verdict with outside_ok False and the error named."""
    try:
        ref = _load(ref_path, "ref")
        full = jax.eval_shape(lambda: ref.create_inputs(jnp.float32))
        abstract = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(shrink_shape(x.shape), x.dtype) if getattr(x, "ndim", 0) else x, full)
        cand = _load(cand_path, "cand")
        closed = jax.make_jaxpr(cand.workload)(*abstract)
        outside, n = walk(closed.jaxpr)
        ok, bad = classify(outside)
        return {"outside_ok": ok, "outside": outside, "outside_disallowed": bad, "pallas_calls": n}
    except BaseException as e:  # includes the alarm's TimeoutError and SystemExit from hostile module code
        return {"outside_ok": False, "outside": {}, "outside_disallowed": [], "pallas_calls": 0, "error": type(e).__name__, "msg": str(e)[-300:]}


def main(argv):
    import os, pathlib, signal, tempfile
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax, jax.numpy as jnp
    if argv[0] != "--batch":
        print("RESULT " + json.dumps(score(argv[0], argv[1], jax, jnp)))
        return
    pack, out_path = json.load(open(argv[1])), argv[2]

    def alarm(*_):
        raise TimeoutError("per-kernel alarm")
    signal.signal(signal.SIGALRM, alarm)
    with open(out_path, "w") as out, tempfile.TemporaryDirectory(dir=".") as d:
        for i, item in enumerate(pack):
            ref_p, cand_p = pathlib.Path(d) / f"ref_{i}.py", pathlib.Path(d) / f"cand_{i}.py"
            ref_p.write_text(item["ref"]); cand_p.write_text(item["cand"])
            signal.alarm(60)
            r = score(str(ref_p), str(cand_p), jax, jnp)
            signal.alarm(0)
            out.write(json.dumps({"id": item["id"], **r}) + "\n"); out.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
