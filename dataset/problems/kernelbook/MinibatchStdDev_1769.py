"""MinibatchStdDev: batch std over dim 0 -> global mean -> broadcast map -> concat on channels."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MinibatchStdDev_1769',
    'batch': 16,
    'channels': 256,
    'height': 128,
    'width': 128,
    'alpha': 1e-8,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    batch_size, _, height, width = x.shape
    y = x - jnp.mean(x, axis=0, keepdims=True)
    y = jnp.sqrt(jnp.mean(y * y, axis=0) + CONFIG['alpha'])
    s = jnp.mean(y).astype(x.dtype)
    stat_map = jnp.broadcast_to(s, (batch_size, 1, height, width))
    return jnp.concatenate([x, stat_map], axis=1)
