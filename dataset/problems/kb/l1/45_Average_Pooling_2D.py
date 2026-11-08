"""2D average pooling (nn.AvgPool2d, kernel 11, stride 11, padding 0)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '45_Average_Pooling_2D',
    'batch_size': 8,
    'channels': 64,
    'height': 512,
    'width': 512,
    'kernel_size': 11,
    'stride': 11,
    'padding': 0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch_size'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride'] if CONFIG['stride'] is not None else k
    p = CONFIG['padding']
    summed = lax.reduce_window(
        x,
        jnp.array(0, dtype=x.dtype),
        lax.add,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding=((0, 0), (0, 0), (p, p), (p, p)),
    )
    # count_include_pad=True (PyTorch default): divide by full window size.
    return summed / jnp.array(k * k, dtype=x.dtype)
