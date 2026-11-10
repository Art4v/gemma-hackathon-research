"""MyMaxPool1dPadSame: constant-zero SAME-style pad on the last dim followed by non-overlapping max pool."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MyMaxPool1dPadSame_6833',
    'channels': 1024,
    'length': 65536,
    'kernel_size': 4,
    'padding_stride': 1,
    'pool_stride': 4,
    'pad_value': 0.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['channels'], CONFIG['length']), dtype)
    return (x,)


def workload(x):
    kernel_size = CONFIG['kernel_size']
    padding_stride = CONFIG['padding_stride']
    pool_stride = CONFIG['pool_stride']

    in_dim = x.shape[-1]
    out_dim = (in_dim + padding_stride - 1) // padding_stride
    p = max(0, (out_dim - 1) * padding_stride + kernel_size - in_dim)
    pad_left = p // 2
    pad_right = p - pad_left

    pad_width = ((0, 0),) * (x.ndim - 1) + ((pad_left, pad_right),)
    net = jnp.pad(x, pad_width, mode='constant', constant_values=CONFIG['pad_value'])

    window = (1,) * (net.ndim - 1) + (kernel_size,)
    strides = (1,) * (net.ndim - 1) + (pool_stride,)
    init = jnp.array(-jnp.inf, net.dtype)
    return lax.reduce_window(net, init, lax.max, window, strides, 'VALID')
