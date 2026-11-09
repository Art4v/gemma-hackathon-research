"""[bf16 variant] SpatialAttention: channel mean/max -> concat -> conv2d(2->1, 3x3, pad 1, no bias) -> sigmoid -> multiply with x."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SpatialAttention_15894',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'padding': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    ks = CONFIG['kernel_size']
    w = jax.random.normal(k_w, (1, 2, ks, ks), dtype) * 0.02  # OIHW, like PyTorch
    return x, w


def workload(x, w):
    avg_out = jnp.mean(x, axis=1, keepdims=True)
    max_out = jnp.max(x, axis=1, keepdims=True)
    y = jnp.concatenate([avg_out, max_out], axis=1)
    y = jnp.transpose(y, (0, 2, 3, 1))
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        y, w,
        window_strides=(1, 1),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = jnp.transpose(y, (0, 3, 1, 2))
    return jax.nn.sigmoid(y) * x
