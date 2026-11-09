"""[bf16 variant] Depthwise 2D convolution (groups=in_channels), square input, square kernel, no bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '82_conv_depthwise_2D_square_input_square_kernel',
    'batch_size': 8,
    'in_channels': 64,
    'height': 512,
    'width': 512,
    'kernel_size': 3,
    'stride': 1,
    'padding': 0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kw = jax.random.split(key)
    B, C = CONFIG['batch_size'], CONFIG['in_channels']
    H, W, K = CONFIG['height'], CONFIG['width'], CONFIG['kernel_size']
    x = jax.random.uniform(kx, (B, C, H, W), dtype=dtype)
    w = jax.random.normal(kw, (K, K, 1, C), dtype=dtype) * 0.02  # HWIO
    return (x, w)


def workload(x, w):
    C = x.shape[1]
    s = CONFIG['stride']
    p = CONFIG['padding']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x_nhwc, w,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=C,
    )
    return jnp.transpose(y, (0, 3, 1, 2))
