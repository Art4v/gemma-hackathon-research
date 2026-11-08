"""3D Max Pooling with dilation."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '43_Max_Pooling_3D',
    'batch_size': 8,
    'channels': 8,
    'dim1': 128,
    'dim2': 128,
    'dim3': 128,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'dilation': 3,
}

def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (
        CONFIG['batch_size'],
        CONFIG['channels'],
        CONFIG['dim1'],
        CONFIG['dim2'],
        CONFIG['dim3'],
    )
    x = jax.random.uniform(key, shape=shape, dtype=dtype)
    return (x,)

def workload(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    p = CONFIG['padding']
    d = CONFIG['dilation']

    if p > 0:
        pad_width = ((0, 0), (0, 0), (p, p), (p, p), (p, p))
        x = jnp.pad(x, pad_width, mode='constant', constant_values=-jnp.inf)

    window_dimensions = (1, 1, k, k, k)
    window_strides = (1, 1, s, s, s)
    window_dilation = (1, 1, d, d, d)

    return lax.reduce_window(
        x,
        -jnp.inf,
        lax.max,
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        padding='VALID',
        window_dilation=window_dilation,
    )
