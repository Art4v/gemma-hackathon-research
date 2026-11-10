"""[bf16 variant] DCGAN generator: Linear -> reshape(512,2,2) -> 4x (ConvTranspose2d + ReLU) -> tanh."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DCGanGenerator_15919',
    'batch': 4096,
    'latent_dim': 2048,
    'fc_channels': 512,
    'fc_spatial': 2,
    # (in_ch, out_ch, kernel, stride, padding, output_padding) per ConvTranspose2d
    'convs': (
        (512, 256, 5, 1, 1, 0),
        (256, 128, 4, 2, 1, 0),
        (128, 64, 4, 2, 1, 0),
        (64, 3, 5, 2, 3, 0),
    ),
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 3 + 2 * len(CONFIG['convs']))
    b, d = CONFIG['batch'], CONFIG['latent_dim']
    fc_out = CONFIG['fc_channels'] * CONFIG['fc_spatial'] ** 2
    x = jax.random.uniform(keys[0], (b, d), dtype)
    fc_w = jax.random.normal(keys[1], (d, fc_out), dtype) * 0.02
    fc_b = jax.random.normal(keys[2], (fc_out,), dtype) * 0.02
    conv_params = []
    for i, (cin, cout, k, _, _, _) in enumerate(CONFIG['convs']):
        w = jax.random.normal(keys[3 + 2 * i], (cin, cout, k, k), dtype) * 0.02
        bias = jax.random.normal(keys[4 + 2 * i], (cout,), dtype) * 0.02
        conv_params += [w, bias]
    return (x, fc_w, fc_b, *conv_params)


def _conv_transpose_nhwc(x, w, b, stride, padding, output_padding):
    # w is PyTorch ConvTranspose2d layout (in, out, kH, kW).
    k = w.shape[2]
    kernel = jnp.transpose(jnp.flip(w, axis=(2, 3)), (2, 3, 0, 1))  # HWIO
    lo = k - 1 - padding
    hi = k - 1 - padding + output_padding
    y = lax.conv_general_dilated(
        x, kernel,
        window_strides=(1, 1),
        padding=((lo, hi), (lo, hi)),
        lhs_dilation=(stride, stride),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return y + b


def workload(x, fc_w, fc_b, *conv_params):
    c, s = CONFIG['fc_channels'], CONFIG['fc_spatial']
    h = x @ fc_w + fc_b
    h = h.reshape(x.shape[0], c, s, s)          # NCHW view, like PyTorch
    h = jnp.transpose(h, (0, 2, 3, 1))          # -> NHWC
    for i, (_, _, _, stride, pad, out_pad) in enumerate(CONFIG['convs']):
        w, b = conv_params[2 * i], conv_params[2 * i + 1]
        h = jax.nn.relu(_conv_transpose_nhwc(h, w, b, stride, pad, out_pad))
    h = jnp.tanh(h)
    return jnp.transpose(h, (0, 3, 1, 2))       # -> NCHW
