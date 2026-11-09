"""[bf16 variant] Conv2d -> multiply by per-channel multiplier -> LeakyReLU -> GELU (exact)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '54_Conv2d_Multiply_LeakyReLU_GELU',
    'batch': 16,
    'in_channels': 64,
    'out_channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kw, kb, km = jax.random.split(key, 4)
    B = CONFIG['batch']
    Ci = CONFIG['in_channels']
    Co = CONFIG['out_channels']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']
    x = jax.random.uniform(kx, (B, Ci, H, W), dtype)
    w = jax.random.normal(kw, (K, K, Ci, Co), dtype) * 0.02
    b = jax.random.normal(kb, (Co,), dtype) * 0.02
    multiplier = jax.random.normal(km, (Co, 1, 1), dtype)
    return x, w, b, multiplier


def workload(x, w, b, multiplier):
    x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
    y = lax.conv_general_dilated(
        x, w,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + b
    y = y * multiplier.reshape(-1)
    y = jnp.where(y >= 0, y, y * CONFIG['negative_slope'])
    y = jax.nn.gelu(y, approximate=False)
    return jnp.transpose(y, (0, 3, 1, 2))  # NHWC -> NCHW
