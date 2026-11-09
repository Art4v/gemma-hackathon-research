"""[bf16 variant] Conv2d(5x5, pad 2, bias) -> sigmoid -> * 256 (simple_decoder)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'simple_decoder_7839',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 5,
    'padding': 2,
    'scale': 256.0,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    c, k = CONFIG['channels'], CONFIG['kernel_size']
    x = jax.random.uniform(kx, (CONFIG['batch'], c, CONFIG['height'], CONFIG['width']), dtype)
    w = jax.random.normal(kw, (c, c, k, k), dtype) * 0.02  # PyTorch OIHW
    b = jax.random.normal(kb, (c,), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    p = CONFIG['padding']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(w, (2, 3, 1, 0))
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(1, 1),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = jax.nn.sigmoid(y + b) * CONFIG['scale']
    return jnp.transpose(y, (0, 3, 1, 2))
