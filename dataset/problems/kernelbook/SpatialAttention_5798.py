"""[bf16 variant] SpatialAttention: channel mean/max -> concat -> 3x3 conv (2->1, no bias) -> sigmoid."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SpatialAttention_5798',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
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
    feat = jnp.concatenate([avg_out, max_out], axis=1)
    feat = jnp.transpose(feat, (0, 2, 3, 1))
    w_hwio = jnp.transpose(w, (2, 3, 1, 0))
    # SAME == padding=k//2 for odd k with stride 1, matching padding=1 for k=3
    y = lax.conv_general_dilated(
        feat, w_hwio, window_strides=(1, 1), padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    return jnp.transpose(jax.nn.sigmoid(y), (0, 3, 1, 2))
