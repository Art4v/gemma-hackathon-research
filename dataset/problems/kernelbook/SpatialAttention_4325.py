"""SpatialAttention: channel mean/max -> concat -> Conv2d(2->1, 7x7) -> sigmoid -> x * gate."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SpatialAttention_4325',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 7,
    'padding': 3,
}


def create_inputs(dtype=jnp.float32):
    kx, kw = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    k = CONFIG['kernel_size']
    w = jax.random.normal(kw, (1, 2, k, k), dtype) * 0.02  # OIHW, like PyTorch
    return x, w


def workload(x, w):
    p = CONFIG['padding']
    avg = jnp.mean(x, axis=1, keepdims=True)
    mx = jnp.max(x, axis=1, keepdims=True)
    pooled = jnp.concatenate([avg, mx], axis=1)          # (B, 2, H, W)
    pooled_nhwc = jnp.transpose(pooled, (0, 2, 3, 1))
    w_hwio = jnp.transpose(w, (2, 3, 1, 0))
    conv = lax.conv_general_dilated(
        pooled_nhwc, w_hwio,
        window_strides=(1, 1),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    gate = jax.nn.sigmoid(jnp.transpose(conv, (0, 3, 1, 2)))  # (B, 1, H, W)
    return x * gate
