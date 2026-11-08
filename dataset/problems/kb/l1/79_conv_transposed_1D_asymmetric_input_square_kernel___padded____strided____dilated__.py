"""ConvTranspose1d (stride, padding, dilation, no bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '79_conv_transposed_1D_asymmetric_input_square_kernel___padded____strided____dilated__',
    'batch_size': 16,
    'in_channels': 32,
    'out_channels': 64,
    'kernel_size': 3,
    'length': 131072,
    'stride': 2,
    'padding': 1,
    'dilation': 2,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw = jax.random.split(key)
    x = jax.random.uniform(kx, (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['length']), dtype)
    w = jax.random.normal(kw, (CONFIG['in_channels'], CONFIG['out_channels'], CONFIG['kernel_size']), dtype) * 0.02
    return x, w


def workload(x, w):
    s = CONFIG['stride']
    p = CONFIG['padding']
    d = CONFIG['dilation']
    K = w.shape[2]
    # PyTorch weight (in, out, K) -> flipped WIO kernel (K, in, out)
    k = jnp.transpose(jnp.flip(w, axis=2), (2, 0, 1))
    pad = d * (K - 1) - p
    x_nwc = jnp.transpose(x, (0, 2, 1))
    y = lax.conv_general_dilated(
        x_nwc, k,
        window_strides=(1,),
        padding=[(pad, pad)],
        lhs_dilation=(s,),
        rhs_dilation=(d,),
        dimension_numbers=('NWC', 'WIO', 'NWC'),
    )
    return jnp.transpose(y, (0, 2, 1))
