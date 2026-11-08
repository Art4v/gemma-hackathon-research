"""Conv3d -> Softmax(dim=channels) -> MaxPool3d -> MaxPool3d."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '6_Conv3d_Softmax_MaxPool_MaxPool',
    'batch_size': 64,
    'in_channels': 3,
    'out_channels': 16,
    'depth': 32,
    'height': 64,
    'width': 64,
    'kernel_size': 3,
    'pool_kernel_size': 2,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    c = CONFIG
    x = jax.random.uniform(
        kx, (c['batch_size'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
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


def workload(x, w, b):
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
