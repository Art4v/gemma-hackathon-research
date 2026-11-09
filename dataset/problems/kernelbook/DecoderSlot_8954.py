"""ConvTranspose2d x6 (66->64 k5s2, 64->64 k5s2 x3, 64->64 k2, 64->4 k2), no activations."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DecoderSlot_8954',
    'batch': 8,
    'in_channels': 66,
    'hidden_channels': 64,
    'out_channels': 4,
    'height': 64,
    'width': 64,
    'kernel_sizes': (5, 5, 5, 5, 2, 2),
    'strides': (2, 2, 2, 2, 1, 1),
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    chans = [c['in_channels']] + [c['hidden_channels']] * 5 + [c['out_channels']]
    keys = jax.random.split(jax.random.key(0), 13)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    params = []
    for i, k in enumerate(c['kernel_sizes']):
        # PyTorch ConvTranspose2d weight layout: (in, out, kH, kW)
        w = jax.random.normal(keys[1 + 2 * i], (chans[i], chans[i + 1], k, k), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (chans[i + 1],), dtype) * 0.02
        params += [w, b]
    return (x, *params)


def _conv_transpose(x, w, b, stride):
    # Transposed conv with padding=0 equals a regular conv on the stride-dilated input,
    # padded by k-1 on each side, using the spatially flipped kernel with in/out as HWIO.
    kh, kw = w.shape[2], w.shape[3]
    kernel = jnp.transpose(jnp.flip(w, (2, 3)), (2, 3, 0, 1))
    y = lax.conv_general_dilated(
        x, kernel,
        window_strides=(1, 1),
        padding=((kh - 1, kh - 1), (kw - 1, kw - 1)),
        lhs_dilation=(stride, stride),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    return y + b


def workload(x, *params):
    out = jnp.transpose(x, (0, 2, 3, 1))
    for i, stride in enumerate(CONFIG['strides']):
        out = _conv_transpose(out, params[2 * i], params[2 * i + 1], stride)
    return jnp.transpose(out, (0, 3, 1, 2))
