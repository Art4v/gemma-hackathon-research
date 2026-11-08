"""3D transposed convolution followed by Swish, GroupNorm, and HardSwish."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '60_ConvTranspose3d_Swish_GroupNorm_HardSwish',
    'batch_size': 64,
    'in_channels': 16,
    'out_channels': 32,
    'depth': 16,
    'height': 32,
    'width': 32,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'groups': 4,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    D = CONFIG['depth']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']

    x = jax.random.uniform(k1, (B, C_in, D, H, W), dtype=dtype)
    conv_weight = jax.random.normal(k2, (C_in, C_out, K, K, K), dtype=dtype) * 0.02
    conv_bias = jnp.zeros((C_out,), dtype=dtype)
    gn_weight = jnp.ones((C_out,), dtype=dtype)
    gn_bias = jnp.zeros((C_out,), dtype=dtype)
    return x, conv_weight, conv_bias, gn_weight, gn_bias


def workload(x, conv_weight, conv_bias, gn_weight, gn_bias):
    s = CONFIG['stride']
    p = CONFIG['padding']
    groups = CONFIG['groups']
    eps = CONFIG['eps']
    kd, kh, kw = conv_weight.shape[2], conv_weight.shape[3], conv_weight.shape[4]

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w = jnp.flip(conv_weight, axis=(2, 3, 4))
    w = jnp.transpose(w, (2, 3, 4, 1, 0))
    pad = ((kd - 1 - p, kd - 1 - p), (kh - 1 - p, kh - 1 - p), (kw - 1 - p, kw - 1 - p))

    out = lax.conv_general_dilated(
        x_ndhwc,
        w,
        window_strides=(1, 1, 1),
        padding=pad,
        lhs_dilation=(s, s, s),
        dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'),
    )
    out = out + conv_bias
    x = jnp.transpose(out, (0, 4, 1, 2, 3))

    x = jax.nn.sigmoid(x) * x

    B, C, D_out, H_out, W_out = x.shape
    x = jnp.reshape(x, (B, groups, C // groups, D_out, H_out, W_out))
    mean = jnp.mean(x, axis=(2, 3, 4, 5), keepdims=True)
    var = jnp.var(x, axis=(2, 3, 4, 5), keepdims=True)
    x = (x - mean) / jnp.sqrt(var + eps)
    x = jnp.reshape(x, (B, C, D_out, H_out, W_out))
    x = x * gn_weight[:, None, None, None] + gn_bias[:, None, None, None]

    x = jax.nn.hard_swish(x)
    return x
