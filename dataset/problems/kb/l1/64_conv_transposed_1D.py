"""ConvTranspose1d (stride 1, no padding, no bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '64_conv_transposed_1D',
    'batch_size': 8,
    'in_channels': 128,
    'out_channels': 128,
    'kernel_size': 3,
    'length': 65536,
    'stride': 1,
    'padding': 0,
    'output_padding': 0,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k1, (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['length']), dtype)
    w = jax.random.normal(
        k2, (CONFIG['in_channels'], CONFIG['out_channels'], CONFIG['kernel_size']), dtype) * 0.02
    return x, w


def workload(x, w):
    # x: (N, Cin, L); w: (Cin, Cout, K) PyTorch ConvTranspose1d layout
    k = w.shape[2]
    stride = CONFIG['stride']
    pad = CONFIG['padding']
    out_pad = CONFIG['output_padding']
    x_nwc = jnp.transpose(x, (0, 2, 1))
    w_wio = jnp.transpose(jnp.flip(w, axis=2), (2, 0, 1))  # (K, Cin, Cout)
    lo = k - 1 - pad
    hi = k - 1 - pad + out_pad
    y = lax.conv_general_dilated(
        x_nwc, w_wio,
        window_strides=(1,),
        padding=[(lo, hi)],
        lhs_dilation=(stride,),
        rhs_dilation=(1,),
        dimension_numbers=('NWC', 'WIO', 'NWC'),
        precision=lax.Precision.HIGHEST,
    )
    return jnp.transpose(y, (0, 2, 1))
