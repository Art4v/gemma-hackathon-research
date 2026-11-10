"""Synthesis_prior_net: (ConvTranspose2d 5x5 s2 + LeakyReLU) x2 -> ConvTranspose2d 3x3 s1 + LeakyReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Synthesis_prior_net_4549',
    'batch': 8,
    'channels': 192,
    'height': 128,
    'width': 128,
    'kernel_sizes': (5, 5, 3),
    'strides': (2, 2, 1),
    'paddings': (2, 2, 1),
    'output_paddings': (1, 1, 0),
    'negative_slope': 0.01,
    'bias_init': 0.01,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 4)
    c = CONFIG['channels']
    x = jax.random.uniform(
        keys[0], (CONFIG['batch'], c, CONFIG['height'], CONFIG['width']), dtype
    )
    params = []
    for k, size in zip(keys[1:], CONFIG['kernel_sizes']):
        w = jax.random.normal(k, (c, c, size, size), dtype) * CONFIG['weight_scale']
        b = jnp.full((c,), CONFIG['bias_init'], dtype)
        params.extend([w, b])
    return (x, *params)


def _deconv(x, w, b, stride, padding, output_padding):
    pads = [
        (w.shape[2] - 1 - padding, w.shape[2] - 1 - padding + output_padding),
        (w.shape[3] - 1 - padding, w.shape[3] - 1 - padding + output_padding),
    ]
    y = lax.conv_general_dilated(
        x,
        jnp.flip(w, axis=(2, 3)),
        window_strides=(1, 1),
        padding=pads,
        lhs_dilation=(stride, stride),
        rhs_dilation=(1, 1),
        dimension_numbers=('NHWC', 'IOHW', 'NHWC'),
    )
    return y + b


def workload(x, w1, b1, w2, b2, w3, b3):
    y = jnp.transpose(x, (0, 2, 3, 1))
    layers = ((w1, b1), (w2, b2), (w3, b3))
    for i, (w, b) in enumerate(layers):
        y = _deconv(
            y, w, b, CONFIG['strides'][i], CONFIG['paddings'][i], CONFIG['output_paddings'][i]
        )
        y = jnp.where(y >= 0, y, CONFIG['negative_slope'] * y)
    return jnp.transpose(y, (0, 3, 1, 2))
