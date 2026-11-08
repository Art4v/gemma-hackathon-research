"""3D convolution followed by channel-wise Softmax and two 3D MaxPool operations."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '6_Conv3d_Softmax_MaxPool_MaxPool',
    'batch_size': 64,
    'in_channels': 16,
    'out_channels': 32,
    'depth': 16,
    'height': 64,
    'width': 64,
    'kernel_size': 3,
    'pool_kernel_size': 2,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    D = CONFIG['depth']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']

    x = jax.random.uniform(k1, (B, C_in, D, H, W), dtype=dtype)
    conv_w = jax.random.normal(k2, (C_out, C_in, K, K, K), dtype=dtype) * 0.02
    conv_b = jax.random.normal(k3, (C_out,), dtype=dtype) * 0.02
    return (x, conv_w, conv_b)


def workload(x, conv_w, conv_b):
    p = CONFIG['pool_kernel_size']
    if isinstance(p, int):
        p_d, p_h, p_w = p, p, p
    else:
        p_d, p_h, p_w = p

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_dhwio = jnp.transpose(conv_w, (2, 3, 4, 1, 0))

    dn = lax.conv_dimension_numbers(
        x_ndhwc.shape, w_dhwio.shape, ('NDHWC', 'DHWIO', 'NDHWC')
    )
    y = lax.conv_general_dilated(
        x_ndhwc,
        w_dhwio,
        window_strides=(1, 1, 1),
        padding='VALID',
        dimension_numbers=dn,
    )
    y = y + conv_b.reshape(1, 1, 1, 1, -1)

    y = jax.nn.softmax(y, axis=-1)

    win = (1, p_d, p_h, p_w, 1)
    y = lax.reduce_window(y, -jnp.inf, lax.max, win, win, 'VALID')
    y = lax.reduce_window(y, -jnp.inf, lax.max, win, win, 'VALID')

    return jnp.transpose(y, (0, 4, 1, 2, 3))
