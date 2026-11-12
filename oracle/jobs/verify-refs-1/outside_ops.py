"""The kernel check's rule: which JAX primitives run OUTSIDE pallas_call in a solution's workload, and may they?

Traced to a jaxpr, every contraction is `dot_general`, every reduction is `reduce_*` and every elementwise op is its own primitive, so
aliases, getattr, `.sum()` or a product stored in a variable can't hide work from the check.

The rule. Outside pallas_call, layout and dtype primitives (ALLOWED) are always fine. Any other primitive is allowed only as
metadata: none of its operands is TAINTED and every operand and result has at most METADATA_MAX_ELEMENTS elements. A value is tainted
if it is a floating (any non-integer, non-bool) input of the workload, or an output of any eqn (pallas_call included) with a tainted
operand; literals never are. So small index work on integer inputs (group offsets, page metadata) passes, and float work on the data
outside the kernel fails however small it is. Dtype does not matter for untainted values: Megablox's make_group_metadata bins int
group offsets with jnp.histogram's float32 edges.

Pure: imports without JAX, so it is tested on fake jaxprs.
"""
import math

# Layout, dtype and placement primitives: moving or re-labelling data, never computing on it. gather is not here: it moves data by
# computed indices (a page gather done in XLA), so it falls under the metadata rule.
ALLOWED = frozenset({"reshape", "transpose", "broadcast_in_dim", "squeeze", "expand_dims", "slice", "dynamic_slice", "dynamic_update_slice",
                     "pad", "concatenate", "convert_element_type", "iota", "copy", "copy_p", "device_put", "rev", "split", "stack",
                     "with_memory_space_constraint", "pjit", "closed_call", "custom_jvp_call", "custom_vjp_call", "custom_vjp_call_jaxpr",
                     "remat", "checkpoint", "jit"})
METADATA_MAX_ELEMENTS = 2 ** 16
_INT_DTYPES = ("int", "uint", "bool")  # str(dtype) prefixes: int4..int64, uint4..uint64, bool
_LOOPS = ("scan", "while")  # a carry feeds back into the body, so one pass of the mapping below is not enough
# Where each one's start indices begin among its operands. Shapes are static at trace time, so a start that is not a literal comes
# from the inputs: data-dependent addressing, a gather by another name. Those occurrences are counted as "<name>(dynamic start)",
# which is not in ALLOWED, so the metadata rule applies; literal starts stay layout.
_DYNAMIC_STARTS = {"dynamic_slice": 1, "dynamic_update_slice": 2}


def _sub_jaxprs(params):
    for v in params.values():
        j = getattr(v, "jaxpr", v)
        if hasattr(j, "eqns"):
            yield j


def walk(jaxpr, outside=None, pallas_calls=0):
    """(primitive counts outside pallas_call, number of pallas_calls). Descends into sub-jaxprs (pjit, scan, ...), never into the kernel."""
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
    """(taint of each outvar, number of pallas_calls), counting into outside and heavy (occurrences that break the metadata rule)."""
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
            # Loops and anything else with sub-jaxprs are conservative: every inner input, and every output, is tainted if any
            # operand is. An untainted loop over int metadata still passes.
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
    """(outside, heavy, pallas_calls): every primitive outside pallas_call with its count, and those occurrences that break the
    metadata rule. jaxpr is the workload's own (its invars are the workload's inputs)."""
    outside, heavy = {}, {}
    _, n = _walk_taint(jaxpr, [_float_input(v) for v in jaxpr.invars], outside, heavy)
    return outside, heavy, n


def classify_avals(outside, heavy, allowed=ALLOWED):
    """(ok, sorted disallowed names, sorted names allowed only as metadata). A primitive outside `allowed` is disallowed if any one
    of its occurrences breaks the metadata rule."""
    extra = [p for p in outside if p not in allowed]
    bad = sorted(p for p in extra if heavy.get(p))
    return not bad, bad, sorted(p for p in extra if not heavy.get(p))
