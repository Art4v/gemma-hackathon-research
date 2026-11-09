"""[bf16 variant] GlobalWeightedAvgPool2d: conv1x1 -> sigmoid -> exp -> spatial normalize -> weight input -> spatial sum."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'GlobalWeightedAvgPool2d_1865',
    'batch': 16,
    'features': 128,
    'height': 128,
    'width': 128,
    'flatten': False,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    n, c, h, w = CONFIG['batch'], CONFIG['features'], CONFIG['height'], CONFIG['width']
    x = jax.random.uniform(kx, (n, c, h, w), dtype)
    conv_w = jax.random.normal(kw, (1, c, 1, 1), dtype) * 0.02
    conv_b = jax.random.normal(kb, (1,), dtype) * 0.02
    return x, conv_w, conv_b


def workload(x, conv_w, conv_b):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(conv_w, (2, 3, 1, 0))
    m = lax.conv_general_dilated(
        x_nhwc, w_hwio, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    ) + conv_b
    m = jnp.exp(jax.nn.sigmoid(m))
    m = m / jnp.sum(m, axis=(1, 2), keepdims=True)
    out = jnp.sum(m * x_nhwc, axis=(1, 2), keepdims=True)
    out = jnp.transpose(out, (0, 3, 1, 2))
    if CONFIG['flatten']:
        return out.reshape(out.shape[0], out.shape[1])
    return out
