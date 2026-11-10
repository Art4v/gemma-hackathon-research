"""MinibatchStdLayer: group reshape, per-group stddev, spatial/channel mean, tile, channel concat."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MinibatchStdLayer_14960',
    'batch': 32,
    'channels': 128,
    'height': 128,
    'width': 128,
    'group_size': 4,
    'eps': 1e-08,
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
    group_size = min(CONFIG['group_size'], n)
    y = jnp.reshape(x, (group_size, -1, c, h, w))
    y = y.astype(jnp.float32)
    y = y - jnp.mean(y, axis=0, keepdims=True)
    y = jnp.mean(y * y, axis=0)
    y = jnp.sqrt(y + CONFIG['eps'])
    y = jnp.mean(y, axis=3, keepdims=True)
    y = jnp.mean(y, axis=2, keepdims=True)
    y = jnp.mean(y, axis=1, keepdims=True)
    y = y.astype(x.dtype)
    y = jnp.tile(y, (group_size, 1, h, w))
    return jnp.concatenate([x, y], axis=1)
