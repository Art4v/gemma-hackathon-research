"""[bf16 variant] Depthwise-separable 2D convolution (depthwise Conv2d followed by pointwise 1x1 Conv2d)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '86_conv_depthwise_separable_2D',
    'batch_size': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'dilation': 1,
}

def create_inputs(dtype=jnp.bfloat16):
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']

    x = jax.random.uniform(k0, (B, C_in, H, W), dtype=dtype)
    dw_weight = jax.random.normal(k1, (C_in, 1, K, K), dtype=dtype) * 0.02
    pw_weight = jax.random.normal(k2, (C_out, C_in, 1, 1), dtype=dtype) * 0.02
    return (x, dw_weight, pw_weight)

def workload(x, dw_weight, pw_weight):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    dilation = CONFIG['dilation']
    in_channels = x.shape[1]

    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    dw_hwio = jnp.transpose(dw_weight, (2, 3, 1, 0))
    pw_hwio = jnp.transpose(pw_weight, (2, 3, 1, 0))

    dn = lax.conv_dimension_numbers(x_nhwc.shape, dw_hwio.shape, ('NHWC', 'HWIO', 'NHWC'))

    out_nhwc = lax.conv_general_dilated(
        x_nhwc,
        dw_hwio,
        window_strides=(stride, stride),
        padding=((padding, padding), (padding, padding)),
        rhs_dilation=(dilation, dilation),
        dimension_numbers=dn,
        feature_group_count=in_channels,
    )

    out_nhwc = lax.conv_general_dilated(
        out_nhwc,
        pw_hwio,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=dn,
    )

    return jnp.transpose(out_nhwc, (0, 3, 1, 2))
