"""[bf16 variant] L2Norm over channels (NCHW): x / (sqrt(sum_c x^2) + eps) * weight[c]."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'L2Norm_13237',
    'batch': 16,
    'channels': 512,
    'height': 64,
    'width': 128,
    'scale': 1.0,
    'eps': 1e-10,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    weight = jnp.full((CONFIG['channels'],), CONFIG['scale'], dtype)
    return x, weight


def workload(x, weight):
    norm = jnp.sqrt(jnp.sum(x * x, axis=1, keepdims=True)) + CONFIG['eps']
    return weight[None, :, None, None] * (x / norm)
