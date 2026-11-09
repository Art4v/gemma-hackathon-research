"""[bf16 variant] HyperpriorSynthesis: ConvTranspose2d(5,s2) -> ReLU -> ConvTranspose2d(5,s2) -> ReLU -> ConvTranspose2d(3,s1)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'HyperpriorSynthesis_6148',
    'batch': 8,
    'N': 320,
    'C': 220,
    'height': 64,
    'width': 64,
    'k_up': 5,
    'stride_up': 2,
    'pad_up': 2,
    'out_pad_up': 1,
    'k_out': 3,
    'stride_out': 1,
    'pad_out': 1,
    'out_pad_out': 0,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(keys[0], (c['batch'], c['N'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(keys[1], (c['N'], c['N'], c['k_up'], c['k_up']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['N'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (c['N'], c['N'], c['k_up'], c['k_up']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['N'],), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (c['N'], c['C'], c['k_out'], c['k_out']), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (c['C'],), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def _conv_transpose(x, w, b, stride, padding, output_padding):
    # x: NHWC; w: PyTorch ConvTranspose2d layout (Cin, Cout, kH, kW).
    kh, kw = w.shape[2], w.shape[3]
    w_hwio = jnp.flip(jnp.transpose(w, (2, 3, 0, 1)), axis=(0, 1))
    pad_h = (kh - 1 - padding, kh - 1 - padding + output_padding)
    pad_w = (kw - 1 - padding, kw - 1 - padding + output_padding)
    y = lax.conv_general_dilated(
        x, w_hwio,
        window_strides=(1, 1),
        padding=[pad_h, pad_w],
        lhs_dilation=(stride, stride),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return y + b


def workload(x, w1, b1, w2, b2, w3, b3):
    c = CONFIG
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = jax.nn.relu(_conv_transpose(h, w1, b1, c['stride_up'], c['pad_up'], c['out_pad_up']))
    h = jax.nn.relu(_conv_transpose(h, w2, b2, c['stride_up'], c['pad_up'], c['out_pad_up']))
    h = _conv_transpose(h, w3, b3, c['stride_out'], c['pad_out'], c['out_pad_out'])
    return jnp.transpose(h, (0, 3, 1, 2))
