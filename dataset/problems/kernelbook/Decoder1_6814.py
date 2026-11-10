"""ReflectionPad2d(1) -> Conv2d(64->3, 3x3, stride 1, valid) + bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Decoder1_6814',
    'batch': 16,
    'in_channels': 64,
    'out_channels': 3,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'stride': 1,
    'pad': 1,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    c = CONFIG
    x = jax.random.uniform(
        k_x, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(
        k_w, (c['out_channels'], c['in_channels'], c['kernel_size'], c['kernel_size']),
        dtype) * 0.02
    b = jax.random.normal(k_b, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    p = CONFIG['pad']
    s = CONFIG['stride']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    x_pad = jnp.pad(x_nhwc, ((0, 0), (p, p), (p, p), (0, 0)), mode='reflect')
    w_hwio = jnp.transpose(w, (2, 3, 1, 0))
    out = lax.conv_general_dilated(
        x_pad, w_hwio,
        window_strides=(s, s),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    out = out + b
    return jnp.transpose(out, (0, 3, 1, 2))
