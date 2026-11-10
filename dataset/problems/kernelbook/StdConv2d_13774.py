"""[bf16 variant] StdConv2d: per-output-channel weight standardization followed by Conv2d with bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'StdConv2d_13774',
    'batch': 16,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
    'eps': 1e-10,
}


def create_inputs(dtype=jnp.bfloat16):
    k = jax.random.key(0)
    kx, kw, kb = jax.random.split(k, 3)
    c = CONFIG
    x = jax.random.uniform(kx, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['out_channels'], c['in_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    m = jnp.mean(w, axis=(1, 2, 3), keepdims=True)
    v = jnp.mean(jnp.square(w - m), axis=(1, 2, 3), keepdims=True)
    w = (w - m) / jnp.sqrt(v + CONFIG['eps'])
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(w, (2, 3, 1, 0))
    p = CONFIG['padding']
    s = CONFIG['stride']
    d = CONFIG['dilation']
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + b
    return jnp.transpose(y, (0, 3, 1, 2))
