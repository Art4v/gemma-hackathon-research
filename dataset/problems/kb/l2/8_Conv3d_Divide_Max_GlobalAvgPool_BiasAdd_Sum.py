"""Conv3d -> divide -> MaxPool3d -> global avg pool -> bias add -> sum over channels."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum',
    'batch_size': 128,
    'in_channels': 8,
    'out_channels': 16,
    'depth': 16,
    'height': 64,
    'width': 64,
    'kernel_size': (3, 3, 3),
    'divisor': 2.0,
    'pool_size': (2, 2, 2),
    'sum_dim': 1,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = jax.random.key(0)
    kx, kw, kb, kb2 = jax.random.split(k, 4)
    x = jax.random.uniform(
        kx, (c['batch_size'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    w = jax.random.normal(
        kw, (c['out_channels'], c['in_channels']) + tuple(c['kernel_size']), dtype) * 0.02
    conv_b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    bias = jax.random.normal(kb2, (c['out_channels'], 1, 1, 1), dtype) * 0.02
    return x, w, conv_b, bias


def workload(x, w, conv_b, bias):
    c = CONFIG
    xh = jnp.transpose(x, (0, 2, 3, 4, 1))            # NDHWC
    wh = jnp.transpose(w, (2, 3, 4, 1, 0))            # DHWIO
    y = lax.conv_general_dilated(
        xh, wh, window_strides=(1, 1, 1), padding='VALID',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'))
    y = y + conv_b
    y = y / c['divisor']
    pd, ph, pw = c['pool_size']
    y = lax.reduce_window(
        y, -jnp.inf, lax.max,
        window_dimensions=(1, pd, ph, pw, 1),
        window_strides=(1, pd, ph, pw, 1),
        padding='VALID')
    y = jnp.mean(y, axis=(1, 2, 3))                    # (N, C)
    y = y + bias.reshape(1, -1)
    y = jnp.sum(y, axis=1)                             # sum over channels
    return y.reshape(-1, 1, 1, 1)
