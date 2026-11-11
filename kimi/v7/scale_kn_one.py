"""Weight-scale one reference module for v7 (self-contained; JAX 0.10.2 dataset venv, CPU).

Multiply every feature-dimension CONFIG int (in/out features, hidden, dim, ...) by ONE integer factor f so the main (largest, non-batched) matmul's
contraction K and output N reach JAXBench k-task sizes: max(K, N) in 4096..16384 and weight K*N*itemsize >= 64 MB (it cannot sit whole in VMEM).
If the inputs then exceed 1.2 GB, divide batch-like ints by an exact power of two g. One common factor keeps every equality (in == out) and divisibility
relation the module relies on; the op chain is unchanged. Rule B = some reduction (sum/max/argmax/...) runs over the matmul's N axis afterwards, so a
kernel that tiles N must keep that axis whole (bn = N) or reduce across tiles.
stdin {"code": ...}  ->  stdout one JSON line (ok, K0, N0, K, N, weight_mb, factor_features, batch_divisor, rule_b, dot_share, bytes_new, code)."""
import json, math, os, re, sys, types
os.environ["JAX_PLATFORMS"] = "cpu"
import jax, numpy as np  # noqa: E402
from jax.extend import core as jcore  # noqa: E402

FEAT = re.compile(r"^(in_features|out_features|hidden|hidden_size|hidden_dim|hidden_features|dim|d_model|in_dim|out_dim|input_size|output_size|"
                  r"n_embd|embed_dim|embedding_dim|ffn_dim|intermediate_size|K|N|in_channels|out_channels|features|num_features|d_ff|"
                  r"input_dim|output_dim|hidden1|hidden2|fc1_units|fc2_units|n_hidden|num_hidden|units|n_in|n_out|nin|nout|"
                  r"state_size|action_size|n_features|in_size|out_size|hid_dim|latent_dim|z_dim|num_inputs|num_outputs|hidden_units|indim|outdim|hidden_in|hidden_out|in_dims|out_dims|num_in|latent_size|feat_dim|mid_size|nb_dim|dim1|dim2|d1|d2|channels|num_channels|nc)$")
BATCHY = re.compile(r"^(batch|batch_size|bs|B|n_batch|num_batch|batches|n_samples|num_samples|samples|M|num_tokens|tokens|n_tokens|seq|seq_len|seqlen|"
                    r"T|S|L|length|rows|num_rows|seq1|seq2)$")
REDUCE = {"reduce_sum", "reduce_max", "reduce_min", "argmax", "argmin", "reduce_prod", "cumsum", "cumlogsumexp", "cummax"}
HBM, TOT_HI = 28e9, 1.2e9


def load(code):
    m = types.ModuleType("ref"); exec(compile(code, "ref.py", "exec"), m.__dict__); return m


def nbytes(m):
    return sum(int(np.prod(l.shape)) * np.dtype(l.dtype).itemsize for l in jax.tree_util.tree_leaves(jax.eval_shape(m.create_inputs)))


def eqns(jaxpr):
    for e in jaxpr.eqns:
        for k in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
            if k in e.params:
                sub = e.params[k]; yield from eqns(sub.jaxpr if hasattr(sub, "jaxpr") else sub); break
        else:
            yield e


def analyze(m):
    """(main non-batched dot as (K, N, itemsize, share), rule_b) from the workload's jaxpr."""
    shp = jax.eval_shape(m.create_inputs)
    shp = shp if isinstance(shp, (tuple, list)) else (shp,)
    jx = jax.make_jaxpr(m.workload)(*shp).jaxpr
    dots, total, reds = [], 0, []
    for e in eqns(jx):
        p = e.primitive.name
        if p == "dot_general":
            a, b = e.invars[0].aval, e.invars[1].aval
            (ca, cb), (ba, bb) = e.params["dimension_numbers"]
            r = e.outvars[0].aval.shape
            K = math.prod(a.shape[d] for d in ca) if ca else 1
            fl = 2 * math.prod(r) * K; total += fl
            if not ba:
                dots.append((fl, K, r[-1] if r else 1, np.dtype(b.dtype).itemsize))
        elif p == "conv_general_dilated":
            r = e.outvars[0].aval.shape; w = e.invars[1].aval.shape; rs = e.params["dimension_numbers"].rhs_spec
            total += 2 * math.prod(r) * w[rs[1]] * math.prod(w[d] for d in rs[2:])
        elif p in REDUCE and e.invars and hasattr(e.invars[0], "aval"):
            av = e.invars[0].aval; axes = e.params.get("axes") or (e.params.get("axis"),)
            if av.shape and (len(av.shape) - 1) in tuple(a for a in axes if a is not None):
                reds.append(av.shape[-1])
    if not dots or not total:
        return None, False
    fl, K, N, isz = max(dots)
    return (K, N, isz, fl / total), (N in reds)


def span(code):
    i = code.find("CONFIG = {")
    if i < 0:
        return None
    d, j = 0, code.index("{", i)
    for k in range(j, len(code)):
        d += code[k] == "{"; d -= code[k] == "}"
        if d == 0:
            return j, k + 1


def patch(code, newvals):
    a, b = span(code); block = code[a:b]
    for key, val in newvals.items():
        pat = re.compile(r"(['\"]" + re.escape(key) + r"['\"]\s*:\s*)(\d+)(?=\s*[,}\n#])")
        if len(pat.findall(block)) != 1:
            return None
        block = pat.sub(lambda mm: mm.group(1) + str(val), block)
    block = re.sub(r"(['\"]name['\"]\s*:\s*['\"])([^'\"]+)(['\"])", lambda mm: mm.group(1) + mm.group(2) + "__kn" + mm.group(3), block, count=1)
    return code[:a] + block + code[b:]


def main():
    code = json.loads(sys.stdin.read())["code"]
    if not span(code):
        print(json.dumps({"ok": False, "reason": "no CONFIG dict"})); return
    m = load(code); cfg = m.CONFIG; b0 = nbytes(m)
    d0, rb0 = analyze(m)
    if not d0:
        print(json.dumps({"ok": False, "reason": "no non-batched matmul"})); return
    K0, N0, isz, share = d0
    ints = {k: v for k, v in cfg.items() if isinstance(v, int) and not isinstance(v, bool) and v > 0}
    feats = {k: v for k, v in ints.items() if FEAT.match(k)}
    base = {"K0": K0, "N0": N0, "dot_share": round(share, 3), "bytes_old": b0, "rule_b": rb0}
    if not feats:
        print(json.dumps({"ok": False, "reason": "no feature-dimension CONFIG key", **base})); return
    for f in range(1, 65):
        newf = {k: v * f for k, v in feats.items()}
        for g in (1, 2, 4, 8, 16, 32, 64, 128):
            newb = {k: v // g for k, v in ints.items() if BATCHY.match(k) and k not in feats and v % g == 0 and v // g >= 1} if g > 1 else {}
            if g > 1 and not newb:
                break
            nc = patch(code, {**newf, **newb})
            if nc is None:
                print(json.dumps({"ok": False, "reason": "a CONFIG key appears more than once", **base})); return
            try:
                m2 = load(nc); b = nbytes(m2)
            except Exception:
                break
            if b > TOT_HI or 3 * b > HBM:
                continue
            try:
                d, rb = analyze(m2)
                jax.eval_shape(m2.workload, *jax.eval_shape(m2.create_inputs))
            except Exception:
                break
            if not d:
                break
            K, N = d[0], d[1]
            if 4096 <= max(K, N) <= 16384 and K * N * d[2] >= 64e6:
                print(json.dumps({"ok": True, **base, "K": K, "N": N, "weight_mb": round(K * N * d[2] / 1e6, 1), "factor_features": f,
                                  "batch_divisor": g, "rule_b": rb, "bytes_new": b, "code": nc}))
                return
            if max(K, N) > 16384:
                print(json.dumps({"ok": False, "reason": f"overshoots: K {K} N {N} at f={f}", **base})); return
            break
    print(json.dumps({"ok": False, "reason": "no factor reaches K/N >= 4096 with weight >= 64MB within 1.2GB inputs", **base}))


main()
