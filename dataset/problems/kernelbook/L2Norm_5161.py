"""L2Norm: channel-wise L2 normalization (sqrt(sum x^2 over C) + eps) followed by per-channel scale."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'L2Norm_5161',
    'batch': 16,
    'n_channels': 512,
    'height': 64,
    'width': 64,
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
    weight = jnp.full((CONFIG['n_channels'],), CONFIG['scale'], dtype)
    return x, weight


def workload(x, weight):
    # eps is added after the sqrt, matching the PyTorch module.
    norm = jnp.sqrt(jnp.sum(x * x, axis=1, keepdims=True)) + CONFIG['eps']
    return weight[None, :, None, None] * (x / norm)
