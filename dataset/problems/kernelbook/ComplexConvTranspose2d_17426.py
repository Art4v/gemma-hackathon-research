"""Complex ConvTranspose2d: four real transposed convolutions (re/im weights on re/im inputs) combined into stacked real and imaginary parts."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ComplexConvTranspose2d_17426',
    'batch': 8,
    'in_channel': 128,
    'out_channel': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 1,
    'padding': 0,
    'output_padding': 0,
    'dilation': 1,
    'unbatched_rank': 4,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 5)
    k = CONFIG['kernel_size']
    in_c = CONFIG['in_channel']
    out_c = CONFIG['out_channel']
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], in_c, CONFIG['height'], CONFIG['width'], 2),
        dtype,
    )
    w_re = jax.random.normal(keys[1], (in_c, out_c, k, k), dtype) * 0.02
    b_re = jax.random.normal(keys[2], (out_c,), dtype) * 0.02
    w_im = jax.random.normal(keys[3], (in_c, out_c, k, k), dtype) * 0.02
    b_im = jax.random.normal(keys[4], (out_c,), dtype) * 0.02
    return x, w_re, b_re, w_im, b_im


def _tconv(x, w, b):
    # x: NCHW; w: (in, out, kH, kW) as stored by torch ConvTranspose2d; b: (out,).
    stride = CONFIG['stride']
    dilation = CONFIG['dilation']
    pad = CONFIG['padding']
    out_pad = CONFIG['output_padding']
    padding = [
        (dilation * (k - 1) - pad, dilation * (k - 1) - pad + out_pad)
        for k in w.shape[2:]
    ]
    kernel = jnp.transpose(jnp.flip(w, axis=(2, 3)), (2, 3, 0, 1))
    y = lax.conv_general_dilated(
        jnp.transpose(x, (0, 2, 3, 1)),
        kernel,
        window_strides=(1, 1),
        padding=padding,
        lhs_dilation=(stride, stride),
        rhs_dilation=(dilation, dilation),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return jnp.transpose(y + b, (0, 3, 1, 2))


def workload(x, w_re, b_re, w_im, b_im):
    unbatched = x.ndim == CONFIG['unbatched_rank']
    if unbatched:
        x = x[None]
    x_r = x[..., 0]
    x_i = x[..., 1]
    real = _tconv(x_r, w_re, b_re) - _tconv(x_i, w_im, b_im)
    imaginary = _tconv(x_i, w_re, b_re) + _tconv(x_r, w_im, b_im)
    output = jnp.stack((real, imaginary), axis=-1)
    return output[0] if unbatched else output
