"""Run gate-verified Pallas kernels on a real TPU at the reference's full size: correctness and speed against XLA (G10).

  python tpu_check.py PACK.json OUT.jsonl [--which dev|train] [--shard I/N] [--limit K] [--cpu-smoke]

Same tolerances as the CPU checker (loose: atol=rtol=1e-2; tight: rtol 1e-2, atol 1e-3 * max|ref|), against the reference run at
matmul precision "highest" so TPU's default bf16-pass matmuls in the reference don't count against the kernel. A compile, VMEM or
runtime failure on the chip is a failure, never re-tiled. Timing: jit, 3 warm-up calls, median of 20, block_until_ready, for both
the kernel and jax.jit(reference) at default precision. Resumable: ids already in OUT are skipped. --cpu-smoke runs in interpret
mode with inputs capped at 256 per dimension, to test the plumbing off-TPU only.
"""
import argparse, importlib.util, inspect, json, os, pathlib, signal, statistics, subprocess, sys, tempfile, time

JAX_PIN = "0.10.2"  # the CPU checker's version (teacher_app.py), so API drift can't masquerade as a kernel failure
ITEM_TIMEOUT_S, WARMUP, REPS = 240, 3, 20
# v6e-1 has 32 GB HBM. Inputs, the XLA output and the kernel output must fit together, so a task runs at full size only when
# 3x its input bytes (from eval_shape, nothing allocated) stay under this; otherwise the row says too_large_for_chip (declared in advance).
HBM_BUDGET_BYTES = 28e9

def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pack"); ap.add_argument("out")
    ap.add_argument("--which", default="dev"); ap.add_argument("--shard", default="0/1"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cpu-smoke", action="store_true"); ap.add_argument("--stack", default="colab-tpu")
    ap.add_argument("--shrink", type=int, default=0, help="cpu-smoke only: divide each dim > 256 by N instead of capping at 256")
    ap.add_argument("--compile-only", action="store_true",
                    help="H3 gate (vi), STUDY2_PLAN 13.3: jit(...).lower(...).compile() at full size on abstract inputs; nothing allocated or timed")
    return ap.parse_args(argv)

# As a script: parse, then pin JAX (or force CPU) before jax is imported. Imported (oracle/tpu_oracle.py): none of that, and main() alone
# installs the pallas_call patch and the alarm handler; the importer sets the platform itself.
a = parse_args() if __name__ == "__main__" else None
if a is not None and not a.cpu_smoke:
    import jax
    if jax.__version__ != JAX_PIN:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"jax[tpu]=={JAX_PIN}"], check=True)
        os.execv(sys.executable, [sys.executable] + sys.argv)
elif a is not None:
    os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np, jax, jax.numpy as jnp
from jax.experimental import pallas as pl

_orig, calls = pl.pallas_call, {"n": 0}
def _counted(*args, **kw):
    calls["n"] += 1
    if a.cpu_smoke:
        kw["interpret"] = True
    return _orig(*args, **kw)

def load(code, name, d):
    p = pathlib.Path(d) / f"{name}.py"; p.write_text(code)
    s = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(s)
    sys.modules[name] = m  # dataclasses resolve string annotations through sys.modules[cls.__module__] (JAXBench 62p)
    s.loader.exec_module(m); return m

def close(want, got, tight):
    w, g = jax.tree_util.tree_leaves(want), jax.tree_util.tree_leaves(got)
    def one(x, y):
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        if x.shape != y.shape:
            return False
        atol = 1e-3 * float(np.abs(x).max()) if tight else 1e-2
        return bool(np.allclose(y, x, rtol=1e-2, atol=atol))
    return len(w) == len(g) and all(one(x, y) for x, y in zip(w, g))

def rel_err(want, got):
    """Max abs error over all outputs, divided by max |reference|: comparable across tasks and with XLA's own default precision."""
    w, g = jax.tree_util.tree_leaves(want), jax.tree_util.tree_leaves(got)
    if len(w) != len(g) or any(np.shape(x) != np.shape(y) for x, y in zip(w, g)):
        return None
    return max(float(np.abs(np.asarray(y, np.float32) - np.asarray(x, np.float32)).max()) / max(float(np.abs(np.asarray(x, np.float32)).max()), 1e-30)
               for x, y in zip(w, g))

def timed(f, inputs):
    for _ in range(WARMUP):
        jax.block_until_ready(f(*inputs))
    ts = []
    for _ in range(REPS):
        t = time.perf_counter(); jax.block_until_ready(f(*inputs)); ts.append(time.perf_counter() - t)
    return statistics.median(ts)

def timed_stats(f, inputs):
    """(median, p10, p90) of REPS calls after WARMUP warm-ups: timed()'s protocol, kept separate because the frozen study imports timed."""
    for _ in range(WARMUP):
        jax.block_until_ready(f(*inputs))
    ts = []
    for _ in range(REPS):
        t = time.perf_counter(); jax.block_until_ready(f(*inputs)); ts.append(time.perf_counter() - t)
    q = statistics.quantiles(ts, n=10, method="inclusive")
    return statistics.median(ts), q[0], q[-1]

class Timeout(Exception):
    pass
def _alarm(*_):
    raise Timeout()

def check(item, ref_code):
    r = {"id": item["id"], "task": item["task"], "stack": a.stack, "chip": jax.devices()[0].device_kind,
         **{k: item[k] for k in ("model", "arm", "teacher") if k in item}}
    with tempfile.TemporaryDirectory() as d:
        stage = "ref"
        try:
            ref = load(ref_code, "ref", d)
            full_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(jax.eval_shape(lambda: ref.create_inputs(jnp.float32)))
                             if hasattr(x, "dtype"))
            r["full_input_bytes"] = int(full_bytes)
            if not a.cpu_smoke and 3 * full_bytes > HBM_BUDGET_BYTES:
                r.update(ran=False, failed_stage="ref", error="too_large_for_chip")
                return r
            # cpu-smoke: build at the cap when the reference supports it (all kb_port baselines do); slicing a full-size build
            # allocates about 6 GB for 24_LogSoftmax and times out. The TPU path is unchanged.
            capped = a.cpu_smoke and "cap" in inspect.signature(ref.create_inputs).parameters
            if capped and a.shrink:
                # post hoc blind-spot check: at the 256 cap many tasks become square, hiding swapped dims. Shrinking each dim > 256 by
                # a.shrink (floored at 256) keeps full-size dims distinct, as the chip sees them. Needs the reference's cap_shape hook.
                ref.cap_shape = lambda shape, cap: tuple(d if d <= 256 else max(256, d // a.shrink) for d in shape)
                inputs = ref.create_inputs(jnp.float32, cap=a.shrink)
                r["inputs_mode"] = f"shrink{a.shrink}"
            else:
                inputs = ref.create_inputs(jnp.float32, cap=256) if capped else ref.create_inputs(jnp.float32)
            if a.cpu_smoke and not r.get("inputs_mode"):
                inputs = tuple(x[tuple(slice(0, min(s, 256)) for s in x.shape)] if hasattr(x, "shape") and x.ndim else x for x in inputs)
            r["shapes"] = [list(x.shape) for x in inputs if hasattr(x, "shape")]
            with jax.default_matmul_precision("highest"):
                want = jax.block_until_ready(jax.jit(ref.workload)(*inputs))
            f_ref = jax.jit(ref.workload)
            xla = f_ref(*inputs)
            r["ref_default_tight"], r["err_xla_default"] = close(want, xla, True), rel_err(want, xla)
            stage = "kernel"
            cand = load(item["code"], "cand", d)
            calls["n"] = 0
            f_c = jax.jit(cand.workload)
            got = jax.block_until_ready(f_c(*inputs))
            r.update(pallas_calls=calls["n"], ran=True, correct=close(want, got, False), tight_correct=close(want, got, True),
                     err_kernel=rel_err(want, got))
            if not a.cpu_smoke:
                stage = "timing"
                r["t_kernel_s"], r["t_xla_s"] = timed(f_c, inputs), timed(f_ref, inputs)
                r["speedup_vs_xla"] = r["t_xla_s"] / r["t_kernel_s"]
        except Timeout:
            r.update(ran=r.get("ran", False), failed_stage=stage, error="Timeout")
        except Exception as e:  # compile, VMEM, lowering and runtime failures on the chip are results, not crashes
            r.update(ran=r.get("ran", False), failed_stage=stage, error=type(e).__name__, msg=str(e)[-400:])
            # declared before any Study 2 run (exploratory study): an HBM out-of-memory, or any failure while building the reference, is the harness's
            # fault, not the kernel's: harness_oom rows are excluded from the hardware column and listed. VMEM errors stay failures.
            m = str(e).lower()
            if stage == "ref" or ("resource_exhausted" in m and "vmem" not in m) or "out of memory" in m and "hbm" in m:
                r["harness_oom"] = True
    return r

def compile_check(item, ref_code):
    """Compile the kernel for the default backend (the TPU on a VM; interpret mode under --cpu-smoke) on jax.eval_shape of the
    reference's full-size inputs. Catches Mosaic and VMEM errors a static lowering misses; never allocates an input."""
    r = {"id": item["id"], "task": item["task"], "stack": a.stack, "chip": jax.devices()[0].device_kind, "mode": "compile_only",
         **{k: item[k] for k in ("model", "arm", "teacher") if k in item}}
    with tempfile.TemporaryDirectory() as d:
        stage = "ref"
        try:
            ref = load(ref_code, "ref", d)
            abstract = jax.eval_shape(lambda: ref.create_inputs(jnp.float32))
            r["full_input_bytes"] = int(sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(abstract) if hasattr(x, "dtype")))
            r["shapes"] = [list(x.shape) for x in jax.tree_util.tree_leaves(abstract) if hasattr(x, "shape")]
            stage = "kernel"
            cand = load(item["code"], "cand", d)
            calls["n"] = 0
            jax.jit(cand.workload).lower(*abstract).compile()
            r.update(compiled=True, pallas_calls=calls["n"])
        except Timeout:
            r.update(compiled=False, failed_stage=stage, error="Timeout")
        except Exception as e:  # Mosaic, VMEM and lowering failures are the result
            r.update(compiled=False, failed_stage=stage, error=type(e).__name__, msg=str(e)[-400:])
    return r

def select(pack, which, shard, limit):
    """(refs, this shard's items): item k of pack[which] goes to shard k % n, then the first `limit` (0 = all)."""
    refs = pack["dev_refs"] if which == "dev" else pack["train_refs"]
    i, n = map(int, shard.split("/"))
    return refs, [x for k, x in enumerate(pack[which]) if k % n == i][: limit or None]

def done_ids(out):
    """Ids already written to OUT, so a rerun resumes instead of repeating them."""
    return {json.loads(l)["id"] for l in out.read_text().splitlines() if l.startswith("{") and '"meta"' not in l} if out.exists() else set()

def main():
    pl.pallas_call = _counted
    signal.signal(signal.SIGALRM, _alarm)
    pack = json.loads(pathlib.Path(a.pack).read_text())
    refs, items = select(pack, a.which, a.shard, a.limit)
    out = pathlib.Path(a.out)
    done = done_ids(out)
    dev0 = jax.devices()[0]
    meta = {"meta": True, "jax": jax.__version__, "platform": dev0.platform, "device_kind": getattr(dev0, "device_kind", "?"),
            "n_devices": len(jax.devices()), "which": a.which, "shard": a.shard, "items": len(items), **({"compile_only": True} if a.compile_only else {})}
    print(json.dumps(meta), flush=True)
    with out.open("a") as f:
        f.write(json.dumps(meta) + "\n")
        for k, item in enumerate(items):
            if item["id"] in done:
                continue
            signal.alarm(ITEM_TIMEOUT_S)
            r = (compile_check if a.compile_only else check)(item, refs[item["task"]])
            signal.alarm(0)
            f.write(json.dumps(r) + "\n"); f.flush()
            print(k, r["id"], r.get("compiled") if a.compile_only else r.get("correct"), r.get("tight_correct"), r.get("speedup_vs_xla"), r.get("error", ""), flush=True)
            jax.clear_caches()

if __name__ == "__main__":
    main()
