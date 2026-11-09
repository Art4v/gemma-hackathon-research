"""[bf16 variant] MaxMin: reshape channels into pairs, max and min over each pair, concat on channel axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MaxMin_1714',
    'batch': 16,
    'channels': 256,
    'height': 128,
    'width': 128,
    'group': 2,
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
    b, c, h, w = x.shape
    g = CONFIG['group']
    y = jnp.reshape(x, (b, c // g, g, h, w))
    maxes = jnp.max(y, axis=2)
    mins = jnp.min(y, axis=2)
    return jnp.concatenate((maxes, mins), axis=1)
