"""[bf16 variant] ConvTranspose2d (square kernel, strided, padded, dilated, no bias) on asymmetric NCHW input."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '81_conv_transposed_2D_asymmetric_input_square_kernel___dilated____padded____strided__',
    'batch_size': 16,
    'in_channels': 128,
    'out_channels': 64,
    'kernel_size': 3,
    'height_in': 64,
    'width_in': 128,
    'stride': 5,
    'padding': 1,
    'dilation': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kw = jax.random.split(key)
    c = CONFIG
    x = jax.random.uniform(
        kx, (c['batch_size'], c['in_channels'], c['height_in'], c['width_in']), dtype=dtype)
    # PyTorch ConvTranspose2d weight layout: (in_channels, out_channels, kH, kW)
    w = jax.random.normal(
        kw, (c['in_channels'], c['out_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    return (x, w)


def workload(x, w):
    s = CONFIG['stride']
    p = CONFIG['padding']
    d = CONFIG['dilation']
    k = w.shape[2]
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    # (in, out, kH, kW) -> flip spatial -> (kH, kW, in, out)
    w_hwio = jnp.transpose(jnp.flip(w, axis=(2, 3)), (2, 3, 0, 1))
    pad = d * (k - 1) - p
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(1, 1),
        padding=((pad, pad), (pad, pad)),
        lhs_dilation=(s, s),
        rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return jnp.transpose(y, (0, 3, 1, 2))
