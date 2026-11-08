"""3D average pooling (AvgPool3d, count_include_pad=True, ceil_mode=False)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '46_Average_Pooling_3D',
    'batch_size': 8,
    'channels': 16,
    'depth': 64,
    'height': 64,
    'width': 128,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch_size'], CONFIG['channels'], CONFIG['depth'],
         CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    p = CONFIG['padding']
    summed = lax.reduce_window(
        x,
        jnp.array(0, dtype=x.dtype),
        lax.add,
        window_dimensions=(1, 1, k, k, k),
        window_strides=(1, 1, s, s, s),
        padding=((0, 0), (0, 0), (p, p), (p, p), (p, p)),
    )
    return summed / jnp.asarray(k * k * k, dtype=x.dtype)
