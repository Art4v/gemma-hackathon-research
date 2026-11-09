"""[bf16 variant] TopKMaxPooling: flatten spatial dims, take top-k per channel, mean of the top-k values."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'TopKMaxPooling_8394',
    'batch': 32,
    'channels': 256,
    'height': 64,
    'width': 64,
    'kmax': 1.0,
}


def _positive_k(k, n):
    # Mirrors TopKMaxPooling.get_positive_k (Python round, i.e. banker's rounding).
    if k <= 0:
        return 0
    if k < 1:
        return round(k * n)
    if k > n:
        return int(n)
    return int(k)


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    b, c, h, w = x.shape
    n = h * w
    k = _positive_k(CONFIG['kmax'], n)
    top, _ = lax.top_k(x.reshape(b, c, n), k)
    return top.sum(axis=2) / k
