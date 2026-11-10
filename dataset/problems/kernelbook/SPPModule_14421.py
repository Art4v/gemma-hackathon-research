"""[bf16 variant] SPPModule: stride-1 max_pool2d at kernels 5, 9, 13, 17 concatenated with the input along the channel axis."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SPPModule_14421',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'num_levels': 4,
    'kernel_step': 4,
    'kernel_offset': 1,
    'stride': 1,
    'channel_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def _max_pool_nchw(x, kernel_size, stride, padding):
    return lax.reduce_window(
        x,
        -jnp.inf,
        lax.max,
        window_dimensions=(1, 1, kernel_size, kernel_size),
        window_strides=(1, 1, stride, stride),
        padding=((0, 0), (0, 0), (padding, padding), (padding, padding)),
    )


def workload(x):
    pooling_layers = [x]
    for i in range(CONFIG['num_levels']):
        kernel_size = CONFIG['kernel_step'] * (i + 1) + CONFIG['kernel_offset']
        padding = (kernel_size - 1) // 2
        pooling_layers.append(_max_pool_nchw(x, kernel_size, CONFIG['stride'], padding))
    return jnp.concatenate(pooling_layers, axis=CONFIG['channel_axis'])
