"""[bf16 variant] SoftHistogram: (x - centers) -> sigmoid(s*(d + delta/2)) - sigmoid(s*(d - delta/2)) -> sum over dim 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SoftHistogram_8586',
    'bins': 4,
    'min_bin_edge': 4,
    'max_bin_edge': 4,
    'sigma': 4,
    'batch': 16,
    'channels': 512,
    'width': 1024,
}


def _delta():
    return float(CONFIG['max_bin_edge'] - CONFIG['min_bin_edge']) / float(CONFIG['bins'])


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    # Axis -2 must equal bins: torch broadcasts centers (bins, 1) against it.
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['bins'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    centers = (float(CONFIG['min_bin_edge'])
               + _delta() * (jnp.arange(CONFIG['bins'], dtype=jnp.float32) + 0.5)).astype(dtype)
    return x, centers


def workload(x, centers):
    sigma = CONFIG['sigma']
    half = _delta() / 2
    d = x[None] - centers[:, None]
    y = jax.nn.sigmoid(sigma * (d + half)) - jax.nn.sigmoid(sigma * (d - half))
    return jnp.sum(y, axis=1)
