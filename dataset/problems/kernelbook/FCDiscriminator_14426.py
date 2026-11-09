"""FCDiscriminator: conv4x4/s2 -> LeakyReLU -> conv4x4/s2 -> LeakyReLU -> conv4x4/s1 -> LeakyReLU -> conv4x4/s1 -> LeakyReLU -> conv4x4/s1 classifier."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'FCDiscriminator_14426',
    'batch': 16,
    'num_classes': 4,
    'height': 512,
    'width': 512,
    'ndf': 64,
    'kernel_size': 4,
    'padding': 1,
    'strides': (2, 2, 1, 1, 1),
    'negative_slope': 0.2,
}


def _channels():
    c, ndf = CONFIG['num_classes'], CONFIG['ndf']
    return [(c, ndf), (ndf, ndf * 2), (ndf * 2, ndf * 4), (ndf * 4, ndf * 4), (ndf * 4, 1)]


def create_inputs(dtype=jnp.float32):
    chans = _channels()
    keys = jax.random.split(jax.random.key(0), 1 + 2 * len(chans))
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['num_classes'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    k = CONFIG['kernel_size']
    params = []
    for i, (cin, cout) in enumerate(chans):
        w = jax.random.normal(keys[1 + 2 * i], (k, k, cin, cout), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (cout,), dtype) * 0.02
        params += [w, b]
    return (x, *params)


def _conv(x, w, b, stride):
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x, w,
        window_strides=(stride, stride),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    return y + b


def workload(x, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5):
    slope = CONFIG['negative_slope']
    strides = CONFIG['strides']
    h = jnp.transpose(x, (0, 2, 3, 1))
    for (w, b), s in zip([(w1, b1), (w2, b2), (w3, b3), (w4, b4)], strides[:4]):
        h = _conv(h, w, b, s)
        h = jnp.where(h >= 0, h, h * slope)
    h = _conv(h, w5, b5, strides[4])
    return jnp.transpose(h, (0, 3, 1, 2))
