"""Decoder_31: ConvTranspose2d(s1) -> LeakyReLU -> ConvTranspose2d(s2,p1) -> LeakyReLU -> ConvTranspose2d(s2,p1)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Decoder_31',
    'batch': 64,
    'z_dim': 128,
    'height': 64,
    'width': 64,
    'c1': 128,
    'c2': 64,
    'c3': 1,
    'kernel_size': 4,
    'stride1': 1,
    'padding1': 0,
    'stride2': 2,
    'padding2': 1,
    'stride3': 2,
    'padding3': 1,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.float32):
    k = CONFIG['kernel_size']
    kx, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['z_dim'], CONFIG['height'], CONFIG['width']), dtype)
    # PyTorch ConvTranspose2d weight layout: (in, out, kH, kW).
    w1 = jax.random.normal(k1, (CONFIG['z_dim'], CONFIG['c1'], k, k), dtype) * 0.02
    w2 = jax.random.normal(k2, (CONFIG['c1'], CONFIG['c2'], k, k), dtype) * 0.02
    w3 = jax.random.normal(k3, (CONFIG['c2'], CONFIG['c3'], k, k), dtype) * 0.02
    return x, w1, w2, w3


def _conv_transpose_nhwc(x, w, stride, padding):
    # Transposed conv == dilated-input conv with flipped kernel and (k-1-p) padding.
    kh, kw = w.shape[2], w.shape[3]
    w_hwio = jnp.transpose(w[:, :, ::-1, ::-1], (2, 3, 0, 1))
    return lax.conv_general_dilated(
        x, w_hwio,
        window_strides=(1, 1),
        padding=[(kh - 1 - padding, kh - 1 - padding), (kw - 1 - padding, kw - 1 - padding)],
        lhs_dilation=(stride, stride),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )


def _lrelu(x):
    return jnp.where(x >= 0, x, x * CONFIG['negative_slope'])


def workload(x, w1, w2, w3):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _lrelu(_conv_transpose_nhwc(h, w1, CONFIG['stride1'], CONFIG['padding1']))
    h = _lrelu(_conv_transpose_nhwc(h, w2, CONFIG['stride2'], CONFIG['padding2']))
    h = _conv_transpose_nhwc(h, w3, CONFIG['stride3'], CONFIG['padding3'])
    return jnp.transpose(h, (0, 3, 1, 2))
