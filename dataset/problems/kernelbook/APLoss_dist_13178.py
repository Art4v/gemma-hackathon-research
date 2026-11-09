"""[bf16 variant] APLoss_dist: sim_to_dist -> 1x1 conv quantizer -> min/clamp histogram -> cumsum precision/recall -> 1 - mAP."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'APLoss_dist_13178',
    'N': 1024,
    'M': 16384,
    'nq': 25,
    'min': 0.0,
    'max': 1.0,
    'dist_offset': 2.001,
    'eps': 1e-16,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['N'], CONFIG['M'])
    x = jax.random.uniform(k1, shape, dtype)
    label = jax.random.uniform(k2, shape, dtype)
    return x, label


def _quantizer_params(dtype):
    # Mirrors the frozen Conv1d(1, 2*nq, 1) weights set in APLoss.__init__.
    nq, lo, hi = CONFIG['nq'], CONFIG['min'], CONFIG['max']
    a = (nq - 1) / (hi - lo)
    k = jnp.arange(nq)
    w1 = jnp.where(k == 0, 0.0, -a).astype(dtype)
    b1 = jnp.where(k == 0, 1.0, a * lo + (nq - k)).astype(dtype)
    w2 = jnp.where(k == nq - 1, 0.0, a).astype(dtype)
    b2 = jnp.where(k == nq - 1, 1.0, (2 - nq + k) - a * lo).astype(dtype)
    return w1, b1, w2, b2


def workload(x, label):
    d = 1.0 - jnp.sqrt(CONFIG['dist_offset'] - 2.0 * x)
    w1, b1, w2, b2 = _quantizer_params(d.dtype)
    dd = d[:, None, :]
    q1 = w1[None, :, None] * dd + b1[None, :, None]
    q2 = w2[None, :, None] * dd + b2[None, :, None]
    q = jnp.maximum(jnp.minimum(q1, q2), 0.0)  # (N, nq, M)
    nbs = jnp.sum(q, axis=-1)
    rec = jnp.sum(q * label[:, None, :], axis=-1)
    prec = jnp.cumsum(rec, axis=-1) / (CONFIG['eps'] + jnp.cumsum(nbs, axis=-1))
    rec = rec / jnp.sum(rec, axis=-1, keepdims=True)
    ap = jnp.sum(prec * rec, axis=-1)
    return 1.0 - jnp.mean(ap)
