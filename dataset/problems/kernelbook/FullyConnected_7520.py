"""Linear (no bias) -> LeakyReLU(0.1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FullyConnected_7520',
    'batch': 8,
    'dim1': 32,
    'dim2': 256,
    'hidden_size': 1024,
    'output_size': 1024,
    'negative_slope': 0.1,
}


def create_inputs(dtype=jnp.float32):
    kx, kw = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['hidden_size']), dtype)
    w = jax.random.normal(kw, (CONFIG['hidden_size'], CONFIG['output_size']), dtype) * 0.02
    return x, w


def workload(x, w):
    y = x @ w
    return jnp.where(y > 0, y, CONFIG['negative_slope'] * y)
