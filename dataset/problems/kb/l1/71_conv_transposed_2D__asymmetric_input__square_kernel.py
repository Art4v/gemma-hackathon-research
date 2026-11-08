"""2D transposed convolution with asymmetric input and square kernel."""
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
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key, 2)
    x = jax.random.uniform(
        k1,
        (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['height_in'], CONFIG['width_in']),
        dtype=dtype,
    )
    weight = jax.random.normal(
        k2,
        (CONFIG['in_channels'], CONFIG['out_channels'] // CONFIG['groups'], CONFIG['kernel_size'], CONFIG['kernel_size']),
        dtype=dtype,
    ) * 0.02
    return (x, weight)

def workload(x, weight):
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    p = CONFIG['padding']
    op = CONFIG['output_padding']
    g = CONFIG['groups']

    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_flipped = jnp.flip(weight, axis=(2, 3))
    if g == 1:
        rhs = jnp.transpose(w_flipped, (2, 3, 0, 1))
    else:
        ic_g = CONFIG['in_channels'] // g
        oc_g = CONFIG['out_channels'] // g
        w_reshaped = jnp.reshape(w_flipped, (g, ic_g, oc_g, k, k))
        w_transposed = jnp.transpose(w_reshaped, (3, 4, 1, 0, 2))
        rhs = jnp.reshape(w_transposed, (k, k, ic_g, CONFIG['out_channels']))

    pad_h = (k - 1 - p, k - 1 - p + op)
    pad_w = (k - 1 - p, k - 1 - p + op)

    out_nhwc = lax.conv_general_dilated(
        lhs=x_nhwc,
        rhs=rhs,
        window_strides=(1, 1),
        padding=(pad_h, pad_w),
        lhs_dilation=(s, s),
         dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=g,
    )
    return jnp.transpose(out_nhwc, (0, 3, 1, 2))
