"""[bf16 variant] L2Norm: channel-wise L2 normalization (sqrt(sum x^2) + eps) followed by per-channel scale."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'L2Norm_7890',
    'batch': 16,
    'channels': 512,
    'height': 64,
    'width': 64,
    'scale': 10.0,
    'eps': 1e-10,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    weight = jnp.full((CONFIG['channels'],), CONFIG['scale'], dtype=dtype)
    return x, weight


def workload(x, weight):
    # eps is added after the sqrt, matching the PyTorch module.
    norm = jnp.sqrt(jnp.sum(x * x, axis=1, keepdims=True)) + CONFIG['eps']
    return weight[None, :, None, None] * (x / norm)
