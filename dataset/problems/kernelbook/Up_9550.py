"""[bf16 variant] ConvTranspose2d (kernel 2, stride 2) followed by ELU activation."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Up_9550',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 2,
    'stride': 2,
    'padding': 0,
    'output_padding': 0,
    'elu_alpha': 1.0,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    weight = jax.random.normal(
        k_w,
        (CONFIG['in_channels'], CONFIG['out_channels'], CONFIG['kernel_size'], CONFIG['kernel_size']),
        dtype,
    ) * CONFIG['weight_scale']
    bias = jax.random.normal(k_b, (CONFIG['out_channels'],), dtype) * CONFIG['weight_scale']
    return x, weight, bias


def workload(x, weight, bias):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    output_padding = CONFIG['output_padding']
    pads = [
        (k - 1 - padding, k - 1 - padding + output_padding)
        for k in weight.shape[2:]
    ]
    nhwc = jnp.transpose(x, (0, 2, 3, 1))
    kernel = jnp.flip(jnp.transpose(weight, (2, 3, 0, 1)), axis=(0, 1))
    out = lax.conv_general_dilated(
        nhwc,
        kernel,
        window_strides=(1, 1),
        padding=pads,
        lhs_dilation=(stride, stride),
        rhs_dilation=(1, 1),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    out = out + bias
    out = jax.nn.elu(out, CONFIG['elu_alpha'])
    return jnp.transpose(out, (0, 3, 1, 2))
