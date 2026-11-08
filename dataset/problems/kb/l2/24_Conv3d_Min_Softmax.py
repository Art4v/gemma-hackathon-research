"""3D convolution followed by min reduction along depth and channel softmax."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '24_Conv3d_Min_Softmax',
    'batch_size': 16,
    'in_channels': 16,
    'out_channels': 64,
    'D': 32,
    'H': 128,
    'W': 128,
    'kernel_size': 3,
    'dim': 2,
    'softmax_axis': 1,
    'stride': 1,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    D, H, W = CONFIG['D'], CONFIG['H'], CONFIG['W']
    K = CONFIG['kernel_size']
    x = jax.random.uniform(k1, (B, C_in, D, H, W), dtype=dtype)
    weight = jax.random.normal(k2, (C_out, C_in, K, K, K), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (C_out,), dtype=dtype) * 0.02
    return (x, weight, bias)


def workload(x, weight, bias):
    s = CONFIG['stride']
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_dhwio = jnp.transpose(weight, (2, 3, 4, 1, 0))
    out = lax.conv_general_dilated(
        x_ndhwc,
        w_dhwio,
        window_strides=(s, s, s),
        padding='VALID',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    out = out + bias
    out = jnp.transpose(out, (0, 4, 1, 2, 3))
    out = jnp.min(out, axis=CONFIG['dim'])
    out = jax.nn.softmax(out, axis=CONFIG['softmax_axis'])
    return out
