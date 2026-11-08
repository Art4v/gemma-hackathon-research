"""Conv1d (dilated, strided, no bias, no padding)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '76_conv_standard_1D_dilated_strided__',
    'batch_size': 16,
    'in_channels': 64,
    'out_channels': 128,
    'kernel_size': 3,
    'length': 65536,
    'stride': 3,
    'dilation': 4,
    'padding': 0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw = jax.random.split(key)
    x = jax.random.uniform(
        kx, (CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['length']), dtype)
    w = jax.random.normal(
        kw, (CONFIG['kernel_size'], CONFIG['in_channels'], CONFIG['out_channels']), dtype) * 0.02
    return x, w


def workload(x, w):
    # x: (N, C_in, L) ; w: (K, C_in, C_out)
    x_nlc = jnp.transpose(x, (0, 2, 1))
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x_nlc, w,
        window_strides=(CONFIG['stride'],),
        padding=((p, p),),
        rhs_dilation=(CONFIG['dilation'],),
        dimension_numbers=('NWC', 'WIO', 'NWC'),
    )
    return jnp.transpose(y, (0, 2, 1))
