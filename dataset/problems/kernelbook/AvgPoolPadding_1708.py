"""AvgPoolPadding: zero-pad channel dimension then 2D average pool (NCHW)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'AvgPoolPadding_1708',
    'batch': 8,
    'channels_in': 128,
    'num_filters': 128,
    'height': 256,
    'width': 256,
    'stride': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels_in'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    num_zeros = CONFIG['num_filters'] - CONFIG['channels_in']
    stride = CONFIG['stride']
    out = jnp.pad(x, ((0, 0), (0, num_zeros), (0, 0), (0, 0)))
    out = jnp.transpose(out, (0, 2, 3, 1))
    out = lax.reduce_window(
        out,
        0.0,
        lax.add,
        (1, stride, stride, 1),
        (1, stride, stride, 1),
        'VALID',
    )
    out = out / (stride * stride)
    return jnp.transpose(out, (0, 3, 1, 2))
