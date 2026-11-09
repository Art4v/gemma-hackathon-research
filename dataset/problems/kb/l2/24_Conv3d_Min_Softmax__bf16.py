"""[bf16 variant] Conv3d -> min over depth -> softmax over channels."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '24_Conv3d_Min_Softmax',
    'batch_size': 64,
    'in_channels': 3,
    'out_channels': 24,
    'D': 32,
    'H': 64,
    'W': 64,
    'kernel_size': 3,
    'dim': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    c = CONFIG
    x = jax.random.uniform(
        kx, (c['batch_size'], c['in_channels'], c['D'], c['H'], c['W']), dtype)
    k = c['kernel_size']
    w = jax.random.normal(
        kw, (k, k, k, c['in_channels'], c['out_channels']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    # NCDHW -> NDHWC
    xh = jnp.transpose(x, (0, 2, 3, 4, 1))
    y = lax.conv_general_dilated(
        xh, w,
        window_strides=(1, 1, 1),
        padding='VALID',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    y = y + b
    # CONFIG['dim'] == 2 is depth in NCDHW, i.e. axis 1 in NDHWC
    ncdhw_to_ndhwc = (0, 4, 1, 2, 3)
    axis = ncdhw_to_ndhwc[CONFIG['dim']]
    y = jnp.min(y, axis=axis)  # (N, H', W', C)
    y = jax.nn.softmax(y, axis=-1)
    return jnp.transpose(y, (0, 3, 1, 2))  # (N, C, H', W')
