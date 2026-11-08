"""3D transposed convolution followed by Softmax and Sigmoid."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '49_ConvTranspose3d_Softmax_Sigmoid',
    'batch_size': 32,
    'in_channels': 64,
    'out_channels': 64,
    'D': 16,
    'H': 32,
    'W': 32,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    b = CONFIG['batch_size']
    c_in = CONFIG['in_channels']
    c_out = CONFIG['out_channels']
    d, h, w = CONFIG['D'], CONFIG['H'], CONFIG['W']
    k = CONFIG['kernel_size']
    x = jax.random.uniform(k1, (b, c_in, d, h, w), dtype=dtype)
    weight = jax.random.normal(k2, (c_in, c_out, k, k, k), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (c_out,), dtype=dtype) * 0.02
    return x, weight, bias


def workload(x, weight, bias):
    s = CONFIG['stride']
    p = CONFIG['padding']
    op = CONFIG['output_padding']
    kd, kh, kw = weight.shape[2], weight.shape[3], weight.shape[4]

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_dhwio = jnp.transpose(weight, (2, 3, 4, 0, 1))
    w_flipped = jnp.flip(w_dhwio, axis=(0, 1, 2))

    pad = [
        (kd - 1 - p, kd - 1 - p + op),
        (kh - 1 - p, kh - 1 - p + op),
        (kw - 1 - p, kw - 1 - p + op),
    ]
    y_ndhwc = lax.conv_general_dilated(
        x_ndhwc,
        w_flipped,
        window_strides=(1, 1, 1),
        padding=pad,
        lhs_dilation=(s, s, s),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    y_ndhwc = y_ndhwc + bias.reshape(1, 1, 1, 1, -1)
    y = jnp.transpose(y_ndhwc, (0, 4, 1, 2, 3))
    y = jax.nn.softmax(y, axis=1)
    y = jax.nn.sigmoid(y)
    return y
