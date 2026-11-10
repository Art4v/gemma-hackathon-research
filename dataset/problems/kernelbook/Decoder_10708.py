"""ConvTranspose2d -> ReLU -> ConvTranspose2d -> Sigmoid decoder (Decoder_10708)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Decoder_10708',
    'batch': 8,
    'latent_channels': 128,
    'height': 256,
    'width': 256,
    'mid_channels': 16,
    'out_channels': 3,
    'kernel_size': (2, 2),
    'stride': (2, 2),
    'padding': (0, 0),
    'output_padding': (0, 0),
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 5)
    kernel = CONFIG['kernel_size']
    latent = CONFIG['latent_channels']
    mid = CONFIG['mid_channels']
    out = CONFIG['out_channels']
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], latent, CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w1 = jax.random.normal(keys[1], (latent, mid) + kernel, dtype) * 0.02
    b1 = jax.random.normal(keys[2], (mid,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (mid, out) + kernel, dtype) * 0.02
    b2 = jax.random.normal(keys[4], (out,), dtype) * 0.02
    return x, w1, b1, w2, b2


def _conv_transpose(x, w, b):
    # x is NHWC; w is PyTorch ConvTranspose layout (in, out, kH, kW).
    kernel = CONFIG['kernel_size']
    stride = CONFIG['stride']
    pad = [
        (k - 1 - p, k - 1 - p + op)
        for k, p, op in zip(kernel, CONFIG['padding'], CONFIG['output_padding'])
    ]
    y = lax.conv_general_dilated(
        x,
        jnp.flip(w, axis=(2, 3)),
        window_strides=(1,) * len(stride),
        padding=pad,
        lhs_dilation=stride,
        rhs_dilation=(1,) * len(stride),
        dimension_numbers=('NHWC', 'IOHW', 'NHWC'),
    )
    return y + b


def workload(x, w1, b1, w2, b2):
    y = jnp.transpose(x, (0, 2, 3, 1))
    y = jax.nn.relu(_conv_transpose(y, w1, b1))
    y = jax.nn.sigmoid(_conv_transpose(y, w2, b2))
    return jnp.transpose(y, (0, 3, 1, 2))
