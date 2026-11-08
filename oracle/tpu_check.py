"""Helpers for tpu_oracle.py: load a module from source, compare outputs, time a function, read a pack.

Imported only by tpu_oracle.init(), after it has chosen the platform, so importing JAX here is safe.
"""
import importlib.util, json, pathlib, statistics, sys, time

import numpy as np, jax

JAX_PIN = "0.10.2"  # the one JAX version the oracle runs on, so an API change can't look like a kernel failure
WARMUP, REPS = 3, 20
# v6e-1 has 32 GB HBM. Inputs, the XLA output and the kernel output must fit together, so a task runs only when 3x its input bytes
# (from eval_shape, nothing allocated) stay under this; otherwise the row says too_large_for_chip.
HBM_BUDGET_BYTES = 28e9
ATOL = RTOL = 1e-2


def load(code, name, d):
    """Import `code` as module `name` from a file in directory `d`. It is registered in sys.modules first: dataclasses resolve
    string annotations through sys.modules[cls.__module__]."""
    p = pathlib.Path(d) / f"{name}.py"; p.write_text(code)
    s = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m); return m


def _f32(x):
    return np.asarray(x, np.float32)


def close(want, got):
    """Every output leaf has the reference's shape and np.allclose(got, want, rtol=1e-2, atol=1e-2)."""
    w, g = jax.tree_util.tree_leaves(want), jax.tree_util.tree_leaves(got)
    return len(w) == len(g) and all(np.shape(x) == np.shape(y) and bool(np.allclose(_f32(y), _f32(x), rtol=RTOL, atol=ATOL))
                                    for x, y in zip(w, g))


def rel_err(want, got):
    """Max abs error over all outputs, divided by max |reference|: comparable across tasks. None if the output structure differs."""
    w, g = jax.tree_util.tree_leaves(want), jax.tree_util.tree_leaves(got)
    if len(w) != len(g) or any(np.shape(x) != np.shape(y) for x, y in zip(w, g)):
        return None
    return max(float(np.abs(_f32(y) - _f32(x)).max()) / max(float(np.abs(_f32(x)).max()), 1e-30) for x, y in zip(w, g))


def timed_stats(f, inputs):
    """(median, p10, p90) of REPS timed calls after WARMUP warm-ups, each waiting for the device to finish."""
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


def select(pack, shard, limit):
    """(refs, this shard's items) from a pack {"refs": {task: code}, "items": [...]}: item k goes to shard k % n, then the first
    `limit` (0 = all)."""
    i, n = map(int, shard.split("/"))
    return pack["refs"], [x for k, x in enumerate(pack["items"]) if k % n == i][: limit or None]


def done_ids(out):
    """Ids already written to OUT, so a rerun resumes instead of repeating them."""
    return {json.loads(l)["id"] for l in out.read_text().splitlines() if l.startswith("{") and '"meta"' not in l} if out.exists() else set()
