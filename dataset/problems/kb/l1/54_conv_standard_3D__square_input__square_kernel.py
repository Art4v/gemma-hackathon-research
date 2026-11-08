"""Standard 3D convolution with cubic input and cubic kernel."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '54_conv_standard_3D__square_input__square_kernel',
    'batch_size': 16,
    'in_channels': 16,
    'out_channels': 64,
    'kernel_size': 3,
    'depth': 64,
    'width': 64,
    'height': 64,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
    'groups': 1,
}

def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    K = CONFIG['kernel_size']
    D = CONFIG['depth']
    W = CONFIG['width']
    H = CONFIG['height']
    G = CONFIG['groups']
    x = jax.random.uniform(k1, (B, C_in, D, W, H), dtype=dtype)
    weight = jax.random.normal(k2, (C_out, C_in // G, K, K, K), dtype=dtype) * 0.02
    return (x, weight)

def workload(x, weight):
    s = CONFIG['stride']
    p = CONFIG['padding']
    d = CONFIG['dilation']
    g = CONFIG['groups']
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_dhwio = jnp.transpose(weight, (2, 3, 4, 1, 0))
    out_ndhwc = lax.conv_general_dilated(
        x_ndhwc,
        w_dhwio,
        window_strides=(s, s, s),
        padding=((p, p), (p, p), (p, p)),
        rhs_dilation=(d, d, d),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        feature_group_count=g,
    )
    return jnp.transpose(out_ndhwc, (0, 4, 1, 2, 3))
