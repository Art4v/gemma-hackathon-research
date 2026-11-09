"""Bottleneck_nobn: relu -> conv1x1 -> relu -> conv3x3(pad 1) -> concat([out, x], channel)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Bottleneck_nobn_10002',
    'batch': 16,
    'in_planes': 128,
    'growth_rate': 32,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'padding': 1,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w1, k_w2 = jax.random.split(jax.random.key(0), 3)
    c_in = CONFIG['in_planes']
    g = CONFIG['growth_rate']
    k = CONFIG['kernel_size']
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], c_in, CONFIG['height'], CONFIG['width']), dtype)
    w1 = jax.random.normal(k_w1, (1, 1, c_in, 4 * g), dtype) * 0.02  # HWIO
    w2 = jax.random.normal(k_w2, (k, k, 4 * g, g), dtype) * 0.02  # HWIO
    return x, w1, w2


def workload(x, w1, w2):
    p = CONFIG['padding']
    dn = ('NHWC', 'HWIO', 'NHWC')
    h = jnp.transpose(x, (0, 2, 3, 1))
    out = lax.conv_general_dilated(
        jax.nn.relu(h), w1, window_strides=(1, 1), padding='VALID',
        dimension_numbers=dn)
    out = lax.conv_general_dilated(
        jax.nn.relu(out), w2, window_strides=(1, 1), padding=((p, p), (p, p)),
        dimension_numbers=dn)
    out = jnp.transpose(out, (0, 3, 1, 2))
    return jnp.concatenate([out, x], axis=1)
