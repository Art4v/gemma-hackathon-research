"""[bf16 variant] Pool_2348: three parallel MaxPool2d (5/9/13, stride 1, same-size padding) concatenated with the input along channels."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Pool_2348',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'kernel_sizes': (5, 9, 13),
    'stride': 1,
    'paddings': (2, 4, 6),
    'concat_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def _max_pool(x, kernel, stride, pad):
    init = jnp.array(-jnp.inf, dtype=x.dtype)
    return lax.reduce_window(
        x,
        init,
        lax.max,
        window_dimensions=(1, 1, kernel, kernel),
        window_strides=(1, 1, stride, stride),
        padding=((0, 0), (0, 0), (pad, pad), (pad, pad)),
    )


def workload(x):
    stride = CONFIG['stride']
    pooled = [
        _max_pool(x, kernel, stride, pad)
        for kernel, pad in zip(CONFIG['kernel_sizes'], CONFIG['paddings'])
    ]
    return jnp.concatenate(pooled + [x], axis=CONFIG['concat_axis'])
