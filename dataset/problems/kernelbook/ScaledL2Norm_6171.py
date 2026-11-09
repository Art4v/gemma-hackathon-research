"""ScaledL2Norm: L2 normalize over channel dim (F.normalize p=2, dim=1) then per-channel scale."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ScaledL2Norm_6171',
    'batch': 16,
    'channels': 512,
    'height': 64,
    'width': 64,
    'initial_scale': 1.0,
    'eps': 1e-12,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    scale = jnp.full((CONFIG['channels'],), CONFIG['initial_scale'], dtype)
    return x, scale


def workload(x, scale):
    norm = jnp.sqrt(jnp.sum(x * x, axis=1, keepdims=True))
    y = x / jnp.maximum(norm, CONFIG['eps'])
    return y * scale[None, :, None, None]
