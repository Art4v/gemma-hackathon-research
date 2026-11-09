"""[bf16 variant] Gated ConvTranspose2d: ConvTranspose2d(x) * sigmoid(ConvTranspose2d(x))."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'GatedConvTranspose_736',
    'batch': 16,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 1,
    'padding': 0,
    'output_padding': 0,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k_x, k_wf, k_bf, k_wg, k_bg = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(k_x, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w_shape = (c['in_channels'], c['out_channels'], c['kernel_size'], c['kernel_size'])
    w_f = jax.random.normal(k_wf, w_shape, dtype) * 0.02
    b_f = jax.random.normal(k_bf, (c['out_channels'],), dtype) * 0.02
    w_g = jax.random.normal(k_wg, w_shape, dtype) * 0.02
    b_g = jax.random.normal(k_bg, (c['out_channels'],), dtype) * 0.02
    return x, w_f, b_f, w_g, b_g


def _conv_transpose(x_nhwc, w, b):
    # ponytail: groups=1 only, matching the KernelBench module.
    k = w.shape[2]
    s, p, op = CONFIG['stride'], CONFIG['padding'], CONFIG['output_padding']
    # torch weight (in, out, kH, kW) -> flipped HWIO for the equivalent direct conv.
    w_hwio = jnp.transpose(jnp.flip(w, axis=(2, 3)), (2, 3, 0, 1))
    lo, hi = k - 1 - p, k - 1 - p + op
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(1, 1),
        padding=((lo, hi), (lo, hi)),
        lhs_dilation=(s, s),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return y + b


def workload(x, w_f, b_f, w_g, b_g):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    f = _conv_transpose(x_nhwc, w_f, b_f)
    g = jax.nn.sigmoid(_conv_transpose(x_nhwc, w_g, b_g))
    return jnp.transpose(f * g, (0, 3, 1, 2))
