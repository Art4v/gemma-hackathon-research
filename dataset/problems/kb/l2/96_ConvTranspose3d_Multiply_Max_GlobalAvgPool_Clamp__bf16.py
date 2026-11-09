"""[bf16 variant] ConvTranspose3d -> scalar multiply -> MaxPool3d -> global AvgPool3d -> clamp."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '96_ConvTranspose3d_Multiply_Max_GlobalAvgPool_Clamp',
    'batch_size': 256,
    'in_channels': 3,
    'out_channels': 16,
    'depth': 16,
    'height': 32,
    'width': 32,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'scale': 0.5,
    'maxpool_kernel_size': 2,
    'clamp_min': 0.0,
    'clamp_max': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    x = jax.random.uniform(
        kx, (c['batch_size'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    k = c['kernel_size']
    w = jax.random.normal(kw, (c['in_channels'], c['out_channels'], k, k, k), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    c = CONFIG
    s = c['stride']
    p = c['padding']
    k = w.shape[2]
    # NCDHW -> NDHWC
    xh = jnp.transpose(x, (0, 2, 3, 4, 1))
    # (I, O, kd, kh, kw) -> flip spatial -> (kd, kh, kw, I, O)
    wk = jnp.transpose(jnp.flip(w, axis=(2, 3, 4)), (2, 3, 4, 0, 1))
    pad = k - 1 - p
    y = lax.conv_general_dilated(
        xh, wk,
        window_strides=(1, 1, 1),
        padding=[(pad, pad)] * 3,
        lhs_dilation=(s, s, s),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    y = y + b.astype(y.dtype)
    y = y * c['scale']
    m = c['maxpool_kernel_size']
    y = lax.reduce_window(
        y, -jnp.inf, lax.max,
        window_dimensions=(1, m, m, m, 1),
        window_strides=(1, m, m, m, 1),
        padding='VALID',
    )
    y = jnp.mean(y, axis=(1, 2, 3))  # (N, C)
    y = jnp.clip(y, c['clamp_min'], c['clamp_max'])
    return y[:, :, None, None, None]
