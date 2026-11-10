"""[bf16 variant] Flatten -> row L2 normalize (detached norm + eps) -> D1 @ D2.T -> squared -> mean (DiffLoss)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DiffLoss_17060',
    'batch': 32,
    'channels': 128,
    'height': 128,
    'width': 128,
    'eps': 1e-06,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    d1 = jax.random.uniform(k1, shape, dtype)
    d2 = jax.random.uniform(k2, shape, dtype)
    return d1, d2


def _normalize_rows(d):
    d = d.reshape(d.shape[0], -1)
    norm = lax.stop_gradient(jnp.linalg.norm(d, ord=2, axis=1, keepdims=True))
    return d / (norm + CONFIG['eps'])


def workload(d1, d2):
    d1_n = _normalize_rows(d1)
    d2_n = _normalize_rows(d2)
    return jnp.mean(jnp.square(d1_n @ d2_n.T))
