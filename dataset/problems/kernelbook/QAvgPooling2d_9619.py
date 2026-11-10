"""[bf16 variant] QAvgPooling2d: 2D average pooling (kernel 3, stride 1, padding 0) on NCHW input with quantization inactive."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'QAvgPooling2d_9619',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'stride': 1,
    'padding': 0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    p = CONFIG['padding']
    summed = lax.reduce_window(
        x,
        jnp.zeros((), x.dtype),
        lax.add,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding=((0, 0), (0, 0), (p, p), (p, p)),
    )
    return summed / (k * k)
