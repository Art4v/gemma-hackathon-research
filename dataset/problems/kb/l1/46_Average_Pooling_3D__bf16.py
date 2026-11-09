"""[bf16 variant] 3D Average Pooling."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '46_Average_Pooling_3D',
    'batch_size': 8,
    'channels': 16,
    'depth': 64,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (
        CONFIG['batch_size'],
        CONFIG['channels'],
        CONFIG['depth'],
        CONFIG['height'],
        CONFIG['width'],
    )
    x = jax.random.uniform(key, shape, dtype=dtype)
    return (x,)

def workload(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride'] if CONFIG['stride'] is not None else k
    p = CONFIG['padding']

    window_dimensions = (1, 1, k, k, k)
    window_strides = (1, 1, s, s, s)
    padding = ((0, 0), (0, 0), (p, p), (p, p), (p, p))

    sum_pooled = lax.reduce_window(
        x,
        jnp.zeros((), dtype=x.dtype),
        lax.add,
        window_dimensions,
        window_strides,
        padding,
    )
    window_size = k * k * k
    return sum_pooled / jnp.asarray(window_size, dtype=x.dtype)
