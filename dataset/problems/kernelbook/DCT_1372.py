"""ConvTranspose2d (k=2, s=2) followed by tanh."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DCT_1372',
    'batch': 16,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 2,
    'stride': 2,
    'padding': 0,
    'output_padding': 0,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(k_x, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    # PyTorch ConvTranspose2d weight layout: (in, out, kH, kW)
    w = jax.random.normal(
        k_w, (c['in_channels'], c['out_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    b = jax.random.normal(k_b, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    s = CONFIG['stride']
    p = CONFIG['padding']
    op = CONFIG['output_padding']
    k = w.shape[2]
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    # Transposed conv == conv over stride-dilated input with flipped kernel, I/O kept as (in, out).
    w_hwio = jnp.transpose(jnp.flip(w, axis=(2, 3)), (2, 3, 0, 1))
    pad = (k - 1 - p, k - 1 - p + op)
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(1, 1),
        padding=(pad, pad),
        lhs_dilation=(s, s),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = jnp.tanh(y + b)
    return jnp.transpose(y, (0, 3, 1, 2))
