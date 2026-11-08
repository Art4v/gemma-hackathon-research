"""3D convolution, 3D max pooling, logsumexp, and ReLU activation."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '43_Conv3d_Max_LogSumExp_ReLU',
    'batch_size': 8,
    'in_channels': 32,
    'out_channels': 64,
    'depth': 32,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'pool_kernel_size': 2,
    'pool_stride': 2,
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
    weight = jax.random.normal(k2, (C_out, C_in, K, K, K), dtype=dtype) * 0.02
    bias = jax.random.normal(k3, (C_out,), dtype=dtype) * 0.02
    return x, weight, bias

def workload(x, weight, bias):
    s = CONFIG['stride']
    p = CONFIG['padding']
    pk = CONFIG['pool_kernel_size']
    ps = CONFIG['pool_stride']

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    dn = lax.conv_dimension_numbers(
        x_ndhwc.shape, weight.shape, ('NDHWC', 'OIDHW', 'NDHWC')
    )
    out = lax.conv_general_dilated(
        x_ndhwc,
        weight,
        window_strides=(s, s, s),
        padding=((p, p), (p, p), (p, p)),
        dimension_numbers=dn,
    )
    out = out + bias.reshape(1, 1, 1, 1, -1)
    out = lax.reduce_window(
        out,
        -jnp.inf,
        lax.max,
        window_dimensions=(1, pk, pk, pk, 1),
        window_strides=(1, ps, ps, ps, 1),
        padding='VALID',
    )
    out = jax.scipy.special.logsumexp(out, axis=-1, keepdims=True)
    out = jax.nn.relu(out)
    return jnp.transpose(out, (0, 4, 1, 2, 3))
