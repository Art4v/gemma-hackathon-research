"""[bf16 variant] Weight-standardized Conv2d: per-filter weight mean/variance normalization, then NCHW convolution plus bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'StdConv2d_5297',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
    'groups': 1,
    'eps': 1e-10,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w = jax.random.normal(
        k_w,
        (
            CONFIG['out_channels'],
            CONFIG['in_channels'] // CONFIG['groups'],
            CONFIG['kernel_size'],
            CONFIG['kernel_size'],
        ),
        dtype,
    ) * CONFIG['weight_scale']
    b = jax.random.normal(k_b, (CONFIG['out_channels'],), dtype) * CONFIG['weight_scale']
    return x, w, b


def workload(x, w, b):
    axes = (1, 2, 3)
    m = jnp.mean(w, axis=axes, keepdims=True)
    v = jnp.var(w, axis=axes, keepdims=True, ddof=0)
    w_std = (w - m) / jnp.sqrt(v + CONFIG['eps'])

    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(w_std, (2, 3, 1, 0))

    s = CONFIG['stride']
    p = CONFIG['padding']
    d = CONFIG['dilation']
    y = lax.conv_general_dilated(
        x_nhwc,
        w_hwio,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=CONFIG['groups'],
    )
    y = y + b
    return jnp.transpose(y, (0, 3, 1, 2))
