"""[bf16 variant] Mfm: Conv2d(3x3, pad 1) -> split channels in half -> elementwise max."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Mfm_12447',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    c = CONFIG
    x = jax.random.uniform(
        k_x, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(
        k_w, (c['kernel_size'], c['kernel_size'], c['in_channels'], 2 * c['out_channels']),
        dtype) * 0.02
    b = jax.random.normal(k_b, (2 * c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    p = CONFIG['padding']
    s = CONFIG['stride']
    out_channels = b.shape[0] // 2
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x_nhwc, w,
        window_strides=(s, s),
        padding=[(p, p), (p, p)],
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    ) + b
    out = jnp.maximum(y[..., :out_channels], y[..., out_channels:])
    return jnp.transpose(out, (0, 3, 1, 2))
