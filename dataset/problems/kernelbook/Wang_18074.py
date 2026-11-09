"""[bf16 variant] Wang EDU scorer: gather relation coeffs, scale rows of x, sum over dim 1, add bias, softmax over last dim."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Wang_18074',
    'batch': 16,
    'channels': 32,
    'height': 256,
    'width': 512,
    'nrels': 128,
    'bias_init': 0.5,
}


def create_inputs(dtype=jnp.bfloat16):
    k_idx, k_x = jax.random.split(jax.random.key(0))
    c = CONFIG
    rel_indices = jax.random.randint(k_idx, (c['height'],), 0, c['nrels'], dtype=jnp.int32)
    x = jax.random.uniform(k_x, (c['batch'], c['channels'], c['height'], c['width']), dtype)
    d = jnp.ones((c['nrels'], 1), dtype).at[0].set(0)
    b = jnp.asarray(c['bias_init'], dtype)
    return rel_indices, x, d, b


def workload(rel_indices, x, d, b):
    # (H, 1) coefficients broadcast over the last two axes of x, one per row.
    rel_coeffs = d[rel_indices]
    ret = jnp.sum(rel_coeffs * x, axis=1) + b
    return jax.nn.softmax(ret, axis=-1)
