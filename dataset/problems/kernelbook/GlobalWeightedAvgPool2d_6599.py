"""GlobalWeightedAvgPool2d: 1x1 conv -> sigmoid -> exp -> spatial normalize -> weight input -> spatial sum."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'GlobalWeightedAvgPool2d_6599',
    'batch': 16,
    'features': 128,
    'height': 128,
    'width': 128,
    'flatten': False,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    b, c, h, w = CONFIG['batch'], CONFIG['features'], CONFIG['height'], CONFIG['width']
    x = jax.random.uniform(k_x, (b, c, h, w), dtype)
    conv_w = jax.random.normal(k_w, (1, 1, c, 1), dtype) * 0.02  # HWIO
    conv_b = jax.random.normal(k_b, (1,), dtype) * 0.02
    return x, conv_w, conv_b


def workload(x, conv_w, conv_b):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    m = lax.conv_general_dilated(
        x_nhwc, conv_w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    ) + conv_b
    m = jnp.transpose(m, (0, 3, 1, 2))  # (B, 1, H, W)
    m = jnp.exp(jax.nn.sigmoid(m))
    m = m / jnp.sum(m, axis=(2, 3), keepdims=True)
    return jnp.sum(m * x, axis=(2, 3), keepdims=not CONFIG['flatten'])
