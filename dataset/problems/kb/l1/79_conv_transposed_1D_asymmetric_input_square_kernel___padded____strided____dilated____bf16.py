"""[bf16 variant] Transposed 1D convolution with asymmetric input, striding, padding, and dilation."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '79_conv_transposed_1D_asymmetric_input_square_kernel___padded____strided____dilated__',
    'batch_size': 16,
    'in_channels': 128,
    'out_channels': 128,
    'kernel_size': 3,
    'length': 65536,
    'stride': 2,
    'padding': 1,
    'dilation': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
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

    x_nlc = jnp.transpose(x, (0, 2, 1))
    w_kio = jnp.transpose(weight, (2, 0, 1))[::-1]

    eff_k = (kernel_size - 1) * dilation + 1
    pad = eff_k - 1 - padding

    out_nlc = lax.conv_transpose(
        x_nlc,
        w_kio,
        strides=(stride,),
        padding=((pad, pad),),
        rhs_dilation=(dilation,),
        dimension_numbers=('NWC', 'WIO', 'NWC'),
        transpose_kernel=False,
    )
    return jnp.transpose(out_nlc, (0, 2, 1))
