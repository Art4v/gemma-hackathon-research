"""InstanceNormLayer: x - mean_hw(x), then divide by sqrt(mean_hw(xc**2) + eps)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'InstanceNormLayer_7450',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'eps': 1e-8,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(jax.random.key(0), shape, dtype)
    return (x,)


def workload(x):
    if x.ndim != 4:
        raise ValueError(f'expected [N, C, H, W] input, got shape {x.shape}')
    xc = x - jnp.mean(x, axis=(2, 3), keepdims=True)
    return xc / jnp.sqrt(jnp.mean(xc ** 2, axis=(2, 3), keepdims=True) + CONFIG['eps'])
