"""[bf16 variant] 2D Max Pooling operation."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '42_Max_Pooling_2D',
    'batch_size': 8,
    'channels': 64,
    'height': 512,
    'width': 512,
    'kernel_size': 4,
    'stride': 1,
    'padding': 1,
    'dilation': 1,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch_size'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype=dtype)
    return (x,)

def workload(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    p = CONFIG['padding']
    d = CONFIG['dilation']

    if p > 0:
        x = jnp.pad(
            x,
            ((0, 0), (0, 0), (p, p), (p, p)),
            mode='constant',
            constant_values=-jnp.inf,
        )

    return lax.reduce_window(
        x,
        init_value=-jnp.inf,
        computation=lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding='VALID',
        window_dilation=(1, 1, d, d),
    )
