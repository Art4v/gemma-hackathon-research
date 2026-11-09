"""[bf16 variant] Minibatch stddev: grouped variance over batch -> sqrt(+eps) -> mean over CHW -> tile -> concat on channels."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MiniBatchStddevLayer_5253',
    'batch': 32,
    'channels': 256,
    'height': 64,
    'width': 64,
    'group_size': 4,
    'eps': 1e-8,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    n, c, h, w = x.shape
    group_size = min(n, CONFIG['group_size'])
    y = jnp.reshape(x, (group_size, -1, c, h, w))
    y = y - jnp.mean(y, axis=0, keepdims=True)
    y = jnp.mean(y * y, axis=0)
    y = jnp.sqrt(y + CONFIG['eps'])
    y = jnp.mean(y, axis=(1, 2, 3), keepdims=True)
    y = jnp.tile(y, (group_size, 1, h, w))
    return jnp.concatenate([x, y.astype(x.dtype)], axis=1)
