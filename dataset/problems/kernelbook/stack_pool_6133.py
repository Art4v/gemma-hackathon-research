"""MaxPool2d(2,s2) -> ReplicationPad(0,1,0,1) -> MaxPool2d(2,s1) -> MaxPool2d(3,s1,p1) -> mean of the three."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'stack_pool_6133',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'pool_a_kernel': 2,
    'pool_a_stride': 2,
    'pool_b_kernel': 2,
    'pool_b_stride': 1,
    'pool_c_kernel': 3,
    'pool_c_stride': 1,
    'pool_c_padding': 1,
    'rep_pad': 1,
    'num_terms': 3.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    return (x,)


def _maxpool(x, k, s, p):
    init = jnp.array(-jnp.inf, x.dtype)
    return lax.reduce_window(
        x, init, lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding=((0, 0), (0, 0), (p, p), (p, p)))


def workload(x):
    r = CONFIG['rep_pad']
    x1 = _maxpool(x, CONFIG['pool_a_kernel'], CONFIG['pool_a_stride'], 0)
    padded = jnp.pad(x1, ((0, 0), (0, 0), (0, r), (0, r)), mode='edge')
    x2 = _maxpool(padded, CONFIG['pool_b_kernel'], CONFIG['pool_b_stride'], 0)
    x3 = _maxpool(x2, CONFIG['pool_c_kernel'], CONFIG['pool_c_stride'], CONFIG['pool_c_padding'])
    return (x1 + x2 + x3) / CONFIG['num_terms']
