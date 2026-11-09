"""[m1 M1_batch variant of kb/l2/74_ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max] ConvTranspose3d -> LeakyReLU -> per-channel multiply -> LeakyReLU -> MaxPool3d, vmapped over a new leading group axis of size 2 with shared weights and depth 8."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '74_ConvTranspose3d_LeakyReLU_Multiply_LeakyReLU_Max_m1_batch',
    'groups': 2,
    'batch_size': 16,
    'in_channels': 64,
    'out_channels': 64,
    'depth': 8,
    'height': 32,
    'width': 32,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
    'negative_slope': 0.2,
    'pool_kernel': 2,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    key = jax.random.key(0)
    kx, kw, kb, km = jax.random.split(key, 4)
    k = c['kernel_size']
    x = jax.random.uniform(
        kx, (c['groups'], c['batch_size'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['in_channels'], c['out_channels'], k, k, k), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    m = jax.random.normal(km, (c['out_channels'], 1, 1, 1), dtype)
    return x, w, b, m


def _leaky(x, slope):
    return jnp.where(x >= 0, x, x * slope)


def _single(x, w, b, m):
    c = CONFIG
    k = w.shape[2]
    s = c['stride']
    p = c['padding']
    op = c['output_padding']
    slope = c['negative_slope']
    pk = c['pool_kernel']

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    # torch (Cin, Cout, kd, kh, kw) -> flipped DHWIO
    w_k = jnp.flip(w, axis=(2, 3, 4))
    w_k = jnp.transpose(w_k, (2, 3, 4, 0, 1))
    pad = [(k - 1 - p, k - 1 - p + op)] * 3
    y = lax.conv_general_dilated(
        x_ndhwc, w_k,
        window_strides=(1, 1, 1),
        padding=pad,
        lhs_dilation=(s, s, s),
        rhs_dilation=(1, 1, 1),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    y = y + b.reshape(1, 1, 1, 1, -1)
    y = _leaky(y, slope)
    y = y * m.reshape(1, 1, 1, 1, -1)
    y = _leaky(y, slope)
    y = lax.reduce_window(
        y, -jnp.inf, lax.max,
        window_dimensions=(1, pk, pk, pk, 1),
        window_strides=(1, pk, pk, pk, 1),
        padding='VALID',
    )
    return jnp.transpose(y, (0, 4, 1, 2, 3))


def workload(x, w, b, m):
    return jax.vmap(_single, in_axes=(0, None, None, None))(x, w, b, m)
