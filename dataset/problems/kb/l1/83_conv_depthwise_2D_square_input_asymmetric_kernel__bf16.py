"""[bf16 variant] Depthwise 2D convolution with square input and asymmetric kernel (K, 1)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '83_conv_depthwise_2D_square_input_asymmetric_kernel',
    'batch_size': 64,
    'in_channels': 8,
    'height': 512,
    'width': 512,
    'kernel_size': 3,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
}

def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    B = CONFIG['batch_size']
    C = CONFIG['in_channels']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']
    x = jax.random.uniform(k1, (B, C, H, W), dtype=dtype)
    weight = jax.random.normal(k2, (C, 1, K, 1), dtype=dtype) * 0.02
    return (x, weight)

def workload(x, weight):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    dilation = CONFIG['dilation']
    in_channels = x.shape[1]

    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(weight, (2, 3, 1, 0))

    out_nhwc = lax.conv_general_dilated(
        lhs=x_nhwc,
        rhs=w_hwio,
        window_strides=(stride, stride),
        padding=[(padding, padding), (padding, padding)],
        rhs_dilation=(dilation, dilation),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=in_channels,
    )
    return jnp.transpose(out_nhwc, (0, 3, 1, 2))
