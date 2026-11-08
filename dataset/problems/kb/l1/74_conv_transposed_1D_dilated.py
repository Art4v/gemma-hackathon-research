"""1D dilated transposed convolution (ConvTranspose1d)."""
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
    'dilation': 3,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    x = jax.random.uniform(
        k1,
        (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['length']),
        dtype=dtype,
    )
    weight = (
        jax.random.normal(
            k2,
            (CONFIG['in_channels'], CONFIG['out_channels'], CONFIG['kernel_size']),
            dtype=dtype,
        )
        * 0.02
    )
    return (x, weight)


def workload(x, weight):
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    dilation = CONFIG['dilation']
    kernel_size = weight.shape[2]

    pad = dilation * (kernel_size - 1) - padding
    x_nhc = jnp.transpose(x, (0, 2, 1))
    w_hio = jnp.transpose(jnp.flip(weight, axis=2), (2, 0, 1))

    out_nhc = lax.conv_general_dilated(
        lhs=x_nhc,
        rhs=w_hio,
        window_strides=(1,),
        padding=[(pad, pad)],
        lhs_dilation=(stride,),
        rhs_dilation=(dilation,),
        dimension_numbers=('NHC', 'HIO', 'NHC'),
    )
    return jnp.transpose(out_nhc, (0, 2, 1))
