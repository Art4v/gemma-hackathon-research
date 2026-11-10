"""MemoryMoCo forward: l_pos = rowwise dot(q, k), l_neg = q @ memory.T, concat, divide by temperature."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MemoryMoCo_15724',
    'batch': 1024,
    'feature_dim': 256,
    'queue_size': 65536,
    'temperature': 0.07,
    'thresh': 0.0,
}


def create_inputs(dtype=jnp.float32):
    n, d, qs = CONFIG['batch'], CONFIG['feature_dim'], CONFIG['queue_size']
    kq, kk, km = jax.random.split(jax.random.key(0), 3)
    q = jax.random.uniform(kq, (n, d), dtype)
    k = jax.random.uniform(kk, (n, d), dtype)
    # Same init as the torch buffer: uniform in [-stdv, stdv].
    stdv = 1.0 / (d / 3) ** 0.5
    memory = jax.random.uniform(km, (qs, d), dtype, minval=-stdv, maxval=stdv)
    return q, k, memory


def workload(q, k, memory):
    l_pos = jnp.sum(q * k, axis=-1, keepdims=True)
    l_neg = jnp.dot(q, memory.T, precision=lax.Precision.HIGHEST)
    thresh = CONFIG['thresh']
    if thresh != 0:
        # Zero out negatives scoring below (l_pos - thresh), as in the torch hard-negative branch.
        l_neg = jnp.where(l_neg - (l_pos - thresh) < 0, 0.0, l_neg).astype(l_neg.dtype)
    out = jnp.concatenate([l_pos, l_neg], axis=1)
    return out / CONFIG['temperature']
