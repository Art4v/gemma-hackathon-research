"""Decoder_10454: Linear+ReLU -> 3x (ConvTranspose2d+ReLU) -> ConvTranspose2d+Sigmoid."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Decoder_10454',
    'batch': 8192,
    'latent_size': 2048,
    'm': 1024,
    'channels': (128, 64, 32, 1),
    'kernels': (5, 5, 6, 6),
    'stride': 2,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 11)
    b, lat, m = CONFIG['batch'], CONFIG['latent_size'], CONFIG['m']
    x = jax.random.uniform(keys[0], (b, lat), dtype)
    fc_w = jax.random.normal(keys[1], (lat, m), dtype) * 0.02
    fc_b = jax.random.normal(keys[2], (m,), dtype) * 0.02
    params = []
    cin = m
    for i, (cout, k) in enumerate(zip(CONFIG['channels'], CONFIG['kernels'])):
        # PyTorch ConvTranspose2d weight layout: (in, out, kH, kW)
        w = jax.random.normal(keys[3 + 2 * i], (cin, cout, k, k), dtype) * 0.02
        bias = jax.random.normal(keys[4 + 2 * i], (cout,), dtype) * 0.02
        params += [w, bias]
        cin = cout
    return (x, fc_w, fc_b, *params)


def _conv_transpose(x, w, b, stride):
    # x: NHWC, w: (in, out, kH, kW). Padding-0 transposed conv as a dilated conv with a flipped kernel.
    kh, kw = w.shape[2], w.shape[3]
    w_hwio = jnp.transpose(w[:, :, ::-1, ::-1], (2, 3, 0, 1))
    y = lax.conv_general_dilated(
        x, w_hwio,
        window_strides=(1, 1),
        padding=[(kh - 1, kh - 1), (kw - 1, kw - 1)],
        lhs_dilation=(stride, stride),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return y + b


def workload(x, fc_w, fc_b, w1, b1, w2, b2, w3, b3, w4, b4):
    s = CONFIG['stride']
    h = jax.nn.relu(x @ fc_w + fc_b)
    h = h.reshape(h.shape[0], 1, 1, h.shape[1])  # NHWC with 1x1 spatial
    h = jax.nn.relu(_conv_transpose(h, w1, b1, s))
    h = jax.nn.relu(_conv_transpose(h, w2, b2, s))
    h = jax.nn.relu(_conv_transpose(h, w3, b3, s))
    out = jax.nn.sigmoid(_conv_transpose(h, w4, b4, s))
    return jnp.transpose(out, (0, 3, 1, 2))  # back to NCHW
