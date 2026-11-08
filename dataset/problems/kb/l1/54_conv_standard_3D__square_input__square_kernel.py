"""3D convolution (NCDHW input, OIDHW weight, kernel 3, stride 1, no padding, no bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '54_conv_standard_3D__square_input__square_kernel',
    'batch_size': 16,
    'in_channels': 8,
    'out_channels': 64,
    'kernel_size': 3,
    'depth': 64,
    'height': 64,
    'width': 64,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w = jax.random.split(jax.random.key(0), 2)
    c = CONFIG
    x = jax.random.uniform(
        k_x,
        (c['batch_size'], c['in_channels'], c['depth'], c['height'], c['width']),
        dtype,
    )
    w = jax.random.normal(
        k_w,
        (c['out_channels'], c['in_channels'], c['kernel_size'], c['kernel_size'], c['kernel_size']),
        dtype,
    ) * 0.02
    return (x, w)


def workload(x, w):
    # x: (N, C_in, D, H, W) like PyTorch; w: (C_out, C_in, kD, kH, kW) like PyTorch.
    s = CONFIG['stride']
    p = CONFIG['padding']
    d = CONFIG['dilation']
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_dhwio = jnp.transpose(w, (2, 3, 4, 1, 0))
    y = lax.conv_general_dilated(
        x_ndhwc,
        w_dhwio,
        window_strides=(s, s, s),
        padding=[(p, p), (p, p), (p, p)],
        rhs_dilation=(d, d, d),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    return jnp.transpose(y, (0, 4, 1, 2, 3))
