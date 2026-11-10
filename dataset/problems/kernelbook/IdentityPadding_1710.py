"""[bf16 variant] IdentityPadding: zero-pad channel axis of NCHW input, then MaxPool2d with kernel 1 and stride."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'IdentityPadding_1710',
    'batch': 8,
    'channels_in': 128,
    'num_filters': 128,
    'height': 256,
    'width': 256,
    'pool_kernel': 1,
    'stride': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels_in'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    num_zeros = CONFIG['num_filters'] - CONFIG['channels_in']
    kernel = CONFIG['pool_kernel']
    stride = CONFIG['stride']
    out = jnp.pad(x, ((0, 0), (0, num_zeros), (0, 0), (0, 0)))
    out = lax.reduce_window(
        out,
        jnp.array(-jnp.inf, out.dtype),
        lax.max,
        window_dimensions=(1, 1, kernel, kernel),
        window_strides=(1, 1, stride, stride),
        padding='VALID',
    )
    return out
