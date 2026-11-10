"""Squeeze-excitation: global avg pool -> 1x1 conv -> swish -> 1x1 conv -> sigmoid -> channel-wise gating of the input."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SE_4955',
    'batch': 8,
    'in_channels': 128,
    'se_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 1,
    'strides': (1, 1),
    'padding': 'VALID',
    'init_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(key, 5)
    in_ch = CONFIG['in_channels']
    se_ch = CONFIG['se_channels']
    ks = CONFIG['kernel_size']
    scale = CONFIG['init_scale']
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], in_ch, CONFIG['height'], CONFIG['width']), dtype
    )
    se1_w = jax.random.normal(k_w1, (se_ch, in_ch, ks, ks), dtype) * scale
    se1_b = jax.random.normal(k_b1, (se_ch,), dtype) * scale
    se2_w = jax.random.normal(k_w2, (in_ch, se_ch, ks, ks), dtype) * scale
    se2_b = jax.random.normal(k_b2, (in_ch,), dtype) * scale
    return x, se1_w, se1_b, se2_w, se2_b


def workload(x, se1_w, se1_b, se2_w, se2_b):
    dims = ('NHWC', 'OIHW', 'NHWC')
    pooled = jnp.mean(x, axis=(2, 3), keepdims=True)
    y = jnp.transpose(pooled, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        y, se1_w, window_strides=CONFIG['strides'], padding=CONFIG['padding'],
        dimension_numbers=dims,
    ) + se1_b
    y = y * jax.nn.sigmoid(y)
    y = lax.conv_general_dilated(
        y, se2_w, window_strides=CONFIG['strides'], padding=CONFIG['padding'],
        dimension_numbers=dims,
    ) + se2_b
    gate = jax.nn.sigmoid(y)
    gate = jnp.transpose(gate, (0, 3, 1, 2))
    return x * gate
