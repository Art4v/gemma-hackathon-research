"""[bf16 variant] Weight-normalized Conv2d: w = g * v / ||v|| (per out channel), then conv2d + bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'WeightNormConv2d_17016',
    'batch': 16,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k_x, k_v, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(k_x, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    v = jax.random.normal(
        k_v, (c['out_channels'], c['in_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    g = jnp.ones((c['out_channels'], 1, 1, 1), dtype)
    bias = jax.random.normal(k_b, (c['out_channels'],), dtype) * 0.02
    return x, v, g, bias


def workload(x, v, g, bias):
    c = CONFIG
    # torch weight_norm(dim=0): L2 norm over every dim except the output-channel dim, no eps.
    norm = jnp.sqrt(jnp.sum(v * v, axis=(1, 2, 3), keepdims=True))
    w = g * (v / norm)
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(w, (2, 3, 1, 0))
    p = c['padding']
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(c['stride'], c['stride']),
        padding=[(p, p), (p, p)],
        rhs_dilation=(c['dilation'], c['dilation']),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + bias
    return jnp.transpose(y, (0, 3, 1, 2))
