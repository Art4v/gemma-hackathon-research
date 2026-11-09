"""[m1 M1_batch variant of kb/l1/86_conv_depthwise_separable_2D] Depthwise 3x3 conv then pointwise 1x1 conv with shared weights, applied over a new leading axis of 4 views with H and W shrunk from 256 to 128."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '86_conv_depthwise_separable_2D_m1_batch',
    'num_views': 4,
    'batch_size': 8,
    'in_channels': 128,
    'out_channels': 256,
    'kernel_size': 3,
    'height': 128,
    'width': 128,
    'stride': 1,
    'padding': 1,
    'dilation': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kd, kp = jax.random.split(key, 3)
    V = CONFIG['num_views']
    B = CONFIG['batch_size']
    C = CONFIG['in_channels']
    O = CONFIG['out_channels']
    K = CONFIG['kernel_size']
    H = CONFIG['height']
    W = CONFIG['width']
    x = jax.random.uniform(kx, (V, B, C, H, W), dtype=dtype)
    w_dw = jax.random.normal(kd, (K, K, 1, C), dtype) * 0.02
    w_pw = jax.random.normal(kp, (C, O), dtype) * 0.02
    return x, w_dw, w_pw


def _separable(x, w_dw, w_pw):
    s = CONFIG['stride']
    p = CONFIG['padding']
    d = CONFIG['dilation']
    C = x.shape[1]
    xh = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        xh, w_dw,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=C,
        precision=lax.Precision.HIGHEST,
    )
    z = jnp.einsum('nhwc,co->nhwo', y, w_pw, precision=lax.Precision.HIGHEST)
    return jnp.transpose(z, (0, 3, 1, 2))


def workload(x, w_dw, w_pw):
    return jax.vmap(_separable, in_axes=(0, None, None))(x, w_dw, w_pw)
