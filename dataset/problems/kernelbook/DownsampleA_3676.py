"""[bf16 variant] DownsampleA_3676: AvgPool2d(kernel=1, stride=2) on NCHW input, then channel-concat with x * 0."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DownsampleA_3676',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 1,
    'stride': 2,
    'zero_factor': 0,
    'channel_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    y = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.reduce_window(y, 0.0, lax.add, (1, k, k, 1), (1, s, s, 1), 'VALID') / (k * k)
    y = jnp.transpose(y, (0, 3, 1, 2))
    return jnp.concatenate((y, y * CONFIG['zero_factor']), axis=CONFIG['channel_axis'])
