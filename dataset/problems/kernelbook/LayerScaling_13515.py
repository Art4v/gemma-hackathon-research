"""LayerScaling: x / sqrt(mean(x^2 over channel group, H, W) + eps)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerScaling_13515',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'group_size': -1,  # -1 means one group spanning all channels
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    n, c, h, w = x.shape
    g = c if CONFIG['group_size'] == -1 else CONFIG['group_size']
    tmp = x.reshape(n, c // g, g, h, w)
    moment2 = jnp.mean(tmp * tmp, axis=(2, 3, 4), keepdims=True)
    out = tmp / jnp.sqrt(moment2 + CONFIG['eps'])
    return out.reshape(n, c, h, w)
