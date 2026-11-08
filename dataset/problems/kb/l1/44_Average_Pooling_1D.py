"""1D Average Pooling."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '44_Average_Pooling_1D',
    'batch_size': 16,
    'in_channels': 128,
    'input_length': 65536,
    'kernel_size': 8,
    'stride': 1,
    'padding': 4,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['input_length']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    p = CONFIG['padding']
    window_sum = lax.reduce_window(
        x,
        jnp.asarray(0, dtype=x.dtype),
        lax.add,
        window_dimensions=(1, 1, k),
        window_strides=(1, 1, s),
        padding=[(0, 0), (0, 0), (p, p)],
    )
    return window_sum / k
