"""ConvTranspose2d (asymmetric input, square kernel, no bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '71_conv_transposed_2D__asymmetric_input__square_kernel',
    'batch_size': 8,
    'in_channels': 32,
    'out_channels': 32,
    'kernel_size': 3,
    'height_in': 512,
    'width_in': 1024,
    'stride': 1,
    'padding': 0,
    'output_padding': 0,
    'groups': 1,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k1,
        (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['height_in'], CONFIG['width_in']),
        dtype=dtype,
    )
    # PyTorch ConvTranspose2d weight layout: (in_channels, out_channels // groups, k, k)
    w = jax.random.normal(
        k2,
        (CONFIG['in_channels'], CONFIG['out_channels'] // CONFIG['groups'],
         CONFIG['kernel_size'], CONFIG['kernel_size']),
        dtype=dtype,
    ) * 0.02
    return x, w


def workload(x, w):
    assert CONFIG['groups'] == 1
    k = w.shape[2]
    s = CONFIG['stride']
    p = CONFIG['padding']
    op = CONFIG['output_padding']
    # NCHW -> NHWC
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    # (in, out, kh, kw) -> flip spatial -> (kh, kw, in, out) HWIO
    w_hwio = jnp.transpose(jnp.flip(w, axis=(2, 3)), (2, 3, 0, 1))
    lo = k - 1 - p
    hi = k - 1 - p + op
    y = lax.conv_general_dilated(
        x_nhwc,
        w_hwio,
        window_strides=(1, 1),
        padding=((lo, hi), (lo, hi)),
        lhs_dilation=(s, s),
        rhs_dilation=(1, 1),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return jnp.transpose(y, (0, 3, 1, 2))
