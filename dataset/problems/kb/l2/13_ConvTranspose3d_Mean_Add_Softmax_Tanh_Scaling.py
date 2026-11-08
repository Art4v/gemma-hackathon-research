"""3D transposed convolution followed by depth mean pooling, bias addition, channel softmax, tanh, and scaling."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling',
    'batch_size': 8,
    'in_channels': 16,
    'out_channels': 64,
    'depth': 32,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'scaling_factor': 2.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    D = CONFIG['depth']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']

    x = jax.random.uniform(k1, (B, C_in, D, H, W), dtype=dtype)
    conv_weight = jax.random.normal(k2, (C_in, C_out, K, K, K), dtype=dtype) * 0.02
    conv_bias = jax.random.normal(k3, (C_out,), dtype=dtype) * 0.02
    bias = jax.random.normal(k4, (1, C_out, 1, 1, 1), dtype=dtype) * 0.02
    return x, conv_weight, conv_bias, bias


def workload(x, conv_weight, conv_bias, bias):
    s = CONFIG['stride']
    p = CONFIG['padding']
    kd, kh, kw = conv_weight.shape[2], conv_weight.shape[3], conv_weight.shape[4]

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_dhwoi = jnp.transpose(jnp.flip(conv_weight, axis=(2, 3, 4)), (2, 3, 4, 1, 0))

    pad_d = kd - 1 - p
    pad_h = kh - 1 - p
    pad_w = kw - 1 - p

    out_ndhwc = lax.conv_general_dilated(
        x_ndhwc,
        w_dhwoi,
        window_strides=(1, 1, 1),
        padding=((pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w)),
        lhs_dilation=(s, s, s),
        dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'),
    )
    out_ndhwc = out_ndhwc + conv_bias.reshape(1, 1, 1, 1, -1)
    out = jnp.transpose(out_ndhwc, (0, 4, 1, 2, 3))

    out = jnp.mean(out, axis=2, keepdims=True)
    out = out + bias
    out = jax.nn.softmax(out, axis=1)
    out = jnp.tanh(out)
    out = out * CONFIG['scaling_factor']
    return out
