"""[bf16 variant] Standard 2D convolution with asymmetric input and square kernel."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '55_conv_standard_2D__asymmetric_input__square_kernel',
    'batch_size': 8,
    'in_channels': 64,
    'out_channels': 128,
    'height': 256,
    'width': 512,
    'kernel_size': 3,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
    'groups': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.uniform(
        k1,
        (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    w = jax.random.normal(
        k2,
        (
            CONFIG['out_channels'],
            CONFIG['in_channels'] // CONFIG['groups'],
            CONFIG['kernel_size'],
            CONFIG['kernel_size'],
        ),
        dtype=dtype,
    ) * 0.02
    return (x, w)


def workload(x, w):
    s = CONFIG['stride']
    p = CONFIG['padding']
    d = CONFIG['dilation']
    g = CONFIG['groups']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(w, (2, 3, 1, 0))
    out_nhwc = lax.conv_general_dilated(
        x_nhwc,
        w_hwio,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        lhs_dilation=(1, 1),
        rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=g,
    )
    return jnp.transpose(out_nhwc, (0, 3, 1, 2))
