"""ConvTranspose2d -> multiply by scalar -> global avg pool -> global avg pool."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '44_ConvTranspose2d_Multiply_GlobalAvgPool_GlobalAvgPool_Mean',
    'batch_size': 16,
    'in_channels': 64,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
    'multiplier': 0.5,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(k1, (c['batch_size'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(k2, (c['in_channels'], c['out_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    b = jax.random.normal(k3, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    c = CONFIG
    k = w.shape[2]
    s = c['stride']
    p = c['padding']
    op = c['output_padding']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    # (Cin, Cout, kH, kW) -> flipped HWIO
    w_hwio = jnp.transpose(jnp.flip(w, axis=(2, 3)), (2, 3, 0, 1))
    lo = k - 1 - p
    hi = k - 1 - p + op
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(1, 1),
        padding=((lo, hi), (lo, hi)),
        lhs_dilation=(s, s),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + b
    y = jnp.transpose(y, (0, 3, 1, 2))
    y = y * c['multiplier']
    y = jnp.mean(y, axis=(2, 3), keepdims=True)
    y = jnp.mean(y, axis=(2, 3), keepdims=True)
    return y
