"""[bf16 variant] Conv1d (NCL, no bias, stride 1, padding 0) -- KernelBench 67_conv_standard_1D."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '67_conv_standard_1D',
    'batch_size': 8,
    'in_channels': 64,
    'out_channels': 128,
    'kernel_size': 3,
    'length': 131072,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k1, (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['length']), dtype)
    w = jax.random.normal(
        k2, (CONFIG['kernel_size'], CONFIG['in_channels'], CONFIG['out_channels']), dtype) * 0.02
    return x, w


def workload(x, w):
    # x: (N, C_in, L) ; w: (K, C_in, C_out)
    x_nlc = jnp.transpose(x, (0, 2, 1))
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x_nlc, w,
        window_strides=(CONFIG['stride'],),
        padding=[(p, p)],
        rhs_dilation=(CONFIG['dilation'],),
        dimension_numbers=('NWC', 'WIO', 'NWC'),
    )
    return jnp.transpose(y, (0, 2, 1))
