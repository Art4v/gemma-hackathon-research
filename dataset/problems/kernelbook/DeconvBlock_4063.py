"""[bf16 variant] ConvTranspose2d (3x3, stride 2, padding 1) -> ReflectionPad2d (right/bottom 1) -> ELU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DeconvBlock_4063',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'output_padding': 0,
    'pad_left': 0,
    'pad_right': 1,
    'pad_top': 0,
    'pad_bottom': 1,
    'elu_alpha': 1.0,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key_x, key_w, key_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        key_x,
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    weight = jax.random.normal(
        key_w,
        (CONFIG['in_channels'], CONFIG['out_channels'], CONFIG['kernel_size'], CONFIG['kernel_size']),
        dtype,
    ) * CONFIG['weight_scale']
    bias = jax.random.normal(key_b, (CONFIG['out_channels'],), dtype) * CONFIG['weight_scale']
    return x, weight, bias


def workload(x, weight, bias):
    low = CONFIG['kernel_size'] - 1 - CONFIG['padding']
    high = low + CONFIG['output_padding']
    stride = CONFIG['stride']

    out = jnp.transpose(x, (0, 2, 3, 1))
    out = lax.conv_general_dilated(
        out,
        jnp.flip(weight, axis=(2, 3)),
        window_strides=(1, 1),
        padding=((low, high), (low, high)),
        lhs_dilation=(stride, stride),
        rhs_dilation=(1, 1),
        dimension_numbers=('NHWC', 'IOHW', 'NHWC'),
    )
    out = out + bias
    out = jnp.pad(
        out,
        (
            (0, 0),
            (CONFIG['pad_top'], CONFIG['pad_bottom']),
            (CONFIG['pad_left'], CONFIG['pad_right']),
            (0, 0),
        ),
        mode='reflect',
    )
    out = jax.nn.elu(out, alpha=CONFIG['elu_alpha'])
    return jnp.transpose(out, (0, 3, 1, 2))
