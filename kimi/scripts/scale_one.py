"""Scale one reference module's CONFIG so its inputs reach JAXBench k-task size (default target 400 MB, accepted 200-800 MB,
3 x inputs <= the 28 GB HBM budget the oracle uses). Only an integer CONFIG value is multiplied, by an integer factor (so every
divisibility relation the module relies on still holds); batch-like keys are tried first. The op chain is untouched.
stdin: {"code": ...}; stdout: one JSON line. Run with the dataset venv (JAX 0.10.2, CPU)."""
import json, os, re, sys, types
os.environ["JAX_PLATFORMS"] = "cpu"
import jax, jax.numpy as jnp, numpy as np  # noqa: E402

LO, HI, TARGET, HBM = 200e6, 800e6, 400e6, 28e9
BATCHY = ["batch", "batch_size", "bs", "B", "n_batch", "num_batch", "batches", "n_samples", "num_samples", "samples", "N", "n",
          "M", "m", "num_tokens", "tokens", "n_tokens", "T", "seq", "seq_len", "seqlen", "S", "L", "length", "rows", "num_rows"]
SKIP = re.compile(r"kernel|stride|pad|dilat|group|head|eps|axis|dim_?k$|^k$|topk|top_k|seed|iter|step|order|num_layers|layers|"
                  r"window|bins|classes|ratio|scale|factor|alpha|beta|temp|size_k|block|tile|chunk", re.I)


def load(code):
    m = types.ModuleType("ref")
    exec(compile(code, "ref.py", "exec"), m.__dict__)
    return m


def nbytes(m):
    leaves = jax.tree_util.tree_leaves(jax.eval_shape(m.create_inputs))
    return sum(int(np.prod(l.shape)) * np.dtype(l.dtype).itemsize for l in leaves)


def config_span(code):
    i = code.find("CONFIG = {")
    if i < 0:
        return None
    d, j = 0, code.index("{", i)
    for k in range(j, len(code)):
        d += code[k] == "{"
        d -= code[k] == "}"
        if d == 0:
            return j, k + 1
    return None


def patch(code, key, new):
    a, b = config_span(code)
    block = code[a:b]
    pat = re.compile(r"(['\"]" + re.escape(key) + r"['\"]\s*:\s*)(\d+)(?=\s*[,}\n#])")
    if len(pat.findall(block)) != 1:
        return None
    block = pat.sub(lambda mm: mm.group(1) + str(new), block)
    block = re.sub(r"(['\"]name['\"]\s*:\s*['\"])([^'\"]+)(['\"])", lambda mm: mm.group(1) + mm.group(2) + "__scale" + mm.group(3), block, count=1)
    return code[:a] + block + code[b:]


def main():
    code = json.loads(sys.stdin.read())["code"]
    m = load(code)
    b0 = nbytes(m)
    cfg = m.CONFIG
    ints = {k: v for k, v in cfg.items() if isinstance(v, int) and not isinstance(v, bool) and v > 0}
    order = [k for k in BATCHY if k in ints] + sorted([k for k in ints if k not in BATCHY and not SKIP.search(k)], key=lambda k: -ints[k])
    tried = []
    for key in order:
        v = ints[key]
        best = None
        for f in sorted(range(2, 129), key=lambda f: abs(f - TARGET / b0)):  # nearest factors to the target first
            new_code = patch(code, key, v * f)
            if new_code is None:
                break
            try:
                m2 = load(new_code)
                b = nbytes(m2)
            except Exception as e:
                tried.append(f"{key}x{f}: {type(e).__name__}")
                break
            if b < 1.5 * b0:  # this key does not drive the input size
                tried.append(f"{key}: no growth")
                break
            if LO <= b <= HI and 3 * b <= HBM:
                best = (f, new_code, b, m2)
                break
            if len(tried) > 40:
                break
        if best is None:
            continue
        f, new_code, b, m2 = best
        try:
            out_shape = jax.eval_shape(m2.workload, *jax.eval_shape(m2.create_inputs))
        except Exception as e:
            tried.append(f"{key}x{f}: workload trace {type(e).__name__}: {str(e)[:120]}")
            continue
        cpu = "skipped"
        if os.environ.get("RUN_CPU") == "1":
            try:
                inp = m2.create_inputs()
                y = jax.jit(m2.workload)(*inp)
                fin = all(bool(jnp.isfinite(l).all()) for l in jax.tree_util.tree_leaves(y) if jnp.issubdtype(l.dtype, jnp.floating))
                cpu = "finite" if fin else "nonfinite"
            except Exception as e:
                cpu = f"error {type(e).__name__}: {str(e)[:150]}"
        print(json.dumps({"ok": cpu in ("finite", "skipped"), "key": key, "old": v, "new": v * f, "factor": f, "bytes_old": b0,
                          "bytes_new": b, "cpu": cpu, "code": new_code}))
        return
    print(json.dumps({"ok": False, "bytes_old": b0, "reason": "no CONFIG key scales the inputs into 200-800 MB", "tried": tried[:12]}))


main()
