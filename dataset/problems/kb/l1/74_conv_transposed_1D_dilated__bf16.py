"""[bf16 variant] ConvTranspose1d (dilated, no bias): NCW input -> transposed 1D convolution -> NCW output."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '74_conv_transposed_1D_dilated',
    'batch_size': 32,
    'in_channels': 32,
    'out_channels': 64,
    'kernel_size': 5,
    'length': 131072,
    'stride': 1,
    'padding': 0,
    'output_padding': 0,
    'dilation': 3,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kw = jax.random.split(key)
    x = jax.random.uniform(
        kx, (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['length']), dtype)
    # PyTorch ConvTranspose1d weight layout: (in_channels, out_channels, K)
    w = jax.random.normal(
        kw, (CONFIG['in_channels'], CONFIG['out_channels'], CONFIG['kernel_size']), dtype) * 0.02
    return x, w


def workload(x, w):
    s = CONFIG['stride']
    p = CONFIG['padding']
    op = CONFIG['output_padding']
    d = CONFIG['dilation']
    k = w.shape[2]
    x_nwc = jnp.transpose(x, (0, 2, 1))                      # (N, L, Cin)
    w_wio = jnp.transpose(jnp.flip(w, axis=2), (2, 0, 1))    # (K, Cin, Cout)
    pad = d * (k - 1) - p
    y = lax.conv_general_dilated(
        x_nwc, w_wio,
        window_strides=(1,),
        padding=[(pad, pad + op)],
        lhs_dilation=(s,),
        rhs_dilation=(d,),
        dimension_numbers=('NWC', 'WIO', 'NWC'),
    )
    return jnp.transpose(y, (0, 2, 1))                       # (N, Cout, Lout)
