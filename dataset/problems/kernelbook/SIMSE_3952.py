"""[bf16 variant] SIMSE: subtract, global sum, square, divide by squared element count."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SIMSE_3952',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key_pred, key_real = jax.random.split(jax.random.key(0))
    pred = jax.random.uniform(key_pred, shape, dtype)
    real = jax.random.uniform(key_real, shape, dtype)
    return pred, real


def workload(pred, real):
    diffs = real - pred
    n = float(diffs.size)
    return jnp.square(jnp.sum(diffs)) / (n * n)
