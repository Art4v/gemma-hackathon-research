"""PKT loss: row L2-normalize -> NaN->0 -> Gram matmul -> (s+1)/2 -> row-normalize -> KL-style mean."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'PKT_3932',
    'n': 8192,
    'd': 4096,
    'eps': 1e-07,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['n'], CONFIG['d'])
    f_s = jax.random.uniform(k1, shape, dtype)
    f_t = jax.random.uniform(k2, shape, dtype)
    return f_s, f_t


def _similarity(x, eps):
    norm = jnp.sqrt(jnp.sum(x ** 2, axis=1, keepdims=True))
    x = x / (norm + eps)
    x = jnp.where(jnp.isnan(x), jnp.zeros_like(x), x)
    s = jnp.matmul(x, x.T, precision=lax.Precision.HIGHEST)
    s = (s + 1.0) / 2.0
    return s / jnp.sum(s, axis=1, keepdims=True)


def workload(f_s, f_t):
    eps = CONFIG['eps']
    model_sim = _similarity(f_s, eps)
    target_sim = _similarity(f_t, eps)
    return jnp.mean(target_sim * jnp.log((target_sim + eps) / (model_sim + eps)))
