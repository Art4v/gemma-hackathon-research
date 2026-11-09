"""[bf16 variant] ConvTranspose3d, MaxPool3d, Softmax, Subtract, Swish, and channel Max."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max',
    'batch_size': 64,
    'in_channels': 16,
    'out_channels': 16,
    'depth': 16,
    'height': 64,
    'width': 64,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
    'pool_kernel_size': 2,
    'pool_stride': 2,
    'pool_padding': 0,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    b = CONFIG['batch_size']
    ic = CONFIG['in_channels']
    oc = CONFIG['out_channels']
    d = CONFIG['depth']
    h = CONFIG['height']
    w = CONFIG['width']
    ks = CONFIG['kernel_size']

    x = jax.random.uniform(k1, (b, ic, d, h, w), dtype=dtype)
    weight = jax.random.normal(k2, (ic, oc, ks, ks, ks), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (oc,), dtype=dtype) * 0.02
    subtract = jax.random.normal(k4, (oc,), dtype=dtype) * 0.02
    return x, weight, bias, subtract


def workload(x, weight, bias, subtract):
    ks = weight.shape[2]
    s = CONFIG['stride']
    p = CONFIG['padding']
    op = CONFIG['output_padding']
    pk = CONFIG['pool_kernel_size']
    ps = CONFIG['pool_stride']
    pp = CONFIG['pool_padding']

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_flipped = jnp.flip(weight, axis=(2, 3, 4))
    w_dhwoi = jnp.transpose(w_flipped, (2, 3, 4, 1, 0))

    pad_before = ks - 1 - p
    pad_after = ks - 1 - p + op
    conv_padding = [(pad_before, pad_after)] * 3

    x_ndhwc = lax.conv_general_dilated(
        x_ndhwc,
        w_dhwoi,
        window_strides=(1, 1, 1),
        padding=conv_padding,
        lhs_dilation=(s, s, s),
        dimension_numbers=('NDHWC', 'DHWOI', 'NDHWC'),
    )
    x_ndhwc = x_ndhwc + bias.reshape(1, 1, 1, 1, -1)

    pool_pads = [(0, 0), (pp, pp), (pp, pp), (pp, pp), (0, 0)]
    x_ndhwc = lax.reduce_window(
        x_ndhwc,
        -jnp.inf,
        lax.max,
        window_dimensions=(1, pk, pk, pk, 1),
        window_strides=(1, ps, ps, ps, 1),
        padding=pool_pads,
    )

    x_ncdhw = jnp.transpose(x_ndhwc, (0, 4, 1, 2, 3))
    x_ncdhw = jax.nn.softmax(x_ncdhw, axis=1)
    x_ncdhw = x_ncdhw - subtract.reshape(1, -1, 1, 1, 1)
    x_ncdhw = jax.nn.sigmoid(x_ncdhw) * x_ncdhw
    out = jnp.max(x_ncdhw, axis=1)
    return out
