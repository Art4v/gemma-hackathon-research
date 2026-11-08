"""Standard 2D convolution with square input, asymmetric kernel, dilation, and padding."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '80_conv_standard_2D_square_input_asymmetric_kernel___dilated____padded__',
    'batch_size': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 512,
    'width': 512,
    'kernel_size': (5, 9),
    'stride': (1, 1),
    'padding': (2, 4),
    'dilation': (2, 3),
}

def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.uniform(
        k1,
        (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    weight = jax.random.normal(
        k2,
        (CONFIG['out_channels'], CONFIG['in_channels'], CONFIG['kernel_size'][0], CONFIG['kernel_size'][1]),
        dtype=dtype,
    ) * 0.02
    return (x, weight)

def workload(x, weight):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(weight, (2, 3, 1, 0))
    pad_h, pad_w = CONFIG['padding']
    out_nhwc = lax.conv_general_dilated(
        lhs=x_nhwc,
        rhs=w_hwio,
        window_strides=CONFIG['stride'],
        padding=[(pad_h, pad_h), (pad_w, pad_w)],
        rhs_dilation=CONFIG['dilation'],
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return jnp.transpose(out_nhwc, (0, 3, 1, 2))
