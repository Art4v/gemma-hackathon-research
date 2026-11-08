"""3D transposed convolution with asymmetric input and asymmetric kernel."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '58_conv_transposed_3D__asymmetric_input__asymmetric_kernel',
    'batch_size': 16,
    'in_channels': 64,
    'out_channels': 32,
    'kernel_size': (3, 5, 7),
    'stride': (1, 1, 1),
    'padding': (0, 0, 0),
    'output_padding': (0, 0, 0),
    'groups': 1,
    'bias': False,
    'depth_in': 16,
    'height_in': 32,
    'width_in': 64,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    G = CONFIG['groups']
    kD, kH, kW = CONFIG['kernel_size']
    D = CONFIG['depth_in']
    H = CONFIG['height_in']
    W = CONFIG['width_in']

    x = jax.random.uniform(k1, (B, C_in, D, H, W), dtype=dtype)
    weight = jax.random.normal(k2, (C_in, C_out // G, kD, kH, kW), dtype=dtype) * 0.02
    return (x, weight)


def workload(x, weight):
    sD, sH, sW = CONFIG['stride']
    pD, pH, pW = CONFIG['padding']
    opD, opH, opW = CONFIG['output_padding']
    kD, kH, kW = weight.shape[2], weight.shape[3], weight.shape[4]

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_flipped = jnp.flip(weight, axis=(2, 3, 4))
    w_dhwio = jnp.transpose(w_flipped, (2, 3, 4, 0, 1))

    pad_d = (kD - 1 - pD, kD - 1 - pD + opD)
    pad_h = (kH - 1 - pH, kH - 1 - pH + opH)
    pad_w = (kW - 1 - pW, kW - 1 - pW + opW)

    out_ndhwc = lax.conv_general_dilated(
        lhs=x_ndhwc,
        rhs=w_dhwio,
        window_strides=(1, 1, 1),
        padding=(pad_d, pad_h, pad_w),
        lhs_dilation=(sD, sH, sW),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        feature_group_count=CONFIG['groups'],
    )
    return jnp.transpose(out_ndhwc, (0, 4, 1, 2, 3))
