"""L2Norm: channel-wise L2 normalization (sum-square, sqrt, add eps, divide) followed by per-channel scale multiply."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'L2Norm_27',
    'batch': 8,
    'n_channels': 128,
    'height': 256,
    'width': 256,
    'scale': 1.0,
    'eps': 1e-10,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['n_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    weight = jnp.ones((CONFIG['n_channels'],), dtype) * CONFIG['scale']
    return x, weight


def workload(x, weight):
    norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=1, keepdims=True)) + CONFIG['eps']
    normalized = x / norm
    return weight[None, :, None, None] * normalized
