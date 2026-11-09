"""AFMS: adaptive_avg_pool1d -> Linear -> sigmoid gate, (x + alpha) * gate."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AFMS_3585',
    'batch': 16,
    'nb_dim': 512,
    'length': 8192,
}


def create_inputs(dtype=jnp.float32):
    b, c, l = CONFIG['batch'], CONFIG['nb_dim'], CONFIG['length']
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, (b, c, l), dtype)
    w = jax.random.normal(kw, (c, c), dtype) * 0.02
    bias = jax.random.normal(kb, (c,), dtype) * 0.02
    alpha = jnp.ones((c, 1), dtype)
    return x, w, bias, alpha


def workload(x, w, bias, alpha):
    y = jnp.mean(x, axis=-1)
    gate = jax.nn.sigmoid(y @ w + bias)[:, :, None]
    return (x + alpha) * gate
