"""[m1 M1_batch variant of kb/l2/6_Conv3d_Softmax_MaxPool_MaxPool] Conv3d -> softmax over channels -> MaxPool3d -> MaxPool3d, vmapped over a new leading group axis of size 4 with shared conv weights and 32x32x32 volumes."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '6_Conv3d_Softmax_MaxPool_MaxPool_m1_batch',
    'groups': 4,
    'batch_size': 64,
    'in_channels': 3,
    'out_channels': 16,
    'depth': 32,
    'height': 32,
    'width': 32,
    'kernel_size': 3,
    'pool_kernel_size': 2,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    c = CONFIG
    x = jax.random.uniform(
        kx, (c['groups'], c['batch_size'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    k = c['kernel_size']
    w = jax.random.normal(kw, (k, k, k, c['in_channels'], c['out_channels']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def _maxpool3d(x, p):
    # x: NDHWC
    return lax.reduce_window(
        x, -jnp.inf, lax.max,
        window_dimensions=(1, p, p, p, 1),
        window_strides=(1, p, p, p, 1),
        padding='VALID')


def _single(x, w, b):
    p = CONFIG['pool_kernel_size']
    x = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
    y = lax.conv_general_dilated(
        x, w,
        window_strides=(1, 1, 1),
        padding='VALID',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'))
    y = y + b
    y = jax.nn.softmax(y, axis=-1)
    y = _maxpool3d(y, p)
    y = _maxpool3d(y, p)
    return jnp.transpose(y, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW


def workload(x, w, b):
    return jax.vmap(_single, in_axes=(0, None, None))(x, w, b)
