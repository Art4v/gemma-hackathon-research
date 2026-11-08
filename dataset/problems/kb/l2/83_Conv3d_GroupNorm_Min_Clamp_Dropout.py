"""3D convolution followed by GroupNorm, minimum, clamp, and dropout."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '83_Conv3d_GroupNorm_Min_Clamp_Dropout',
    'batch_size': 128,
    'in_channels': 3,
    'out_channels': 16,
    'depth': 16,
    'height': 64,
    'width': 64,
    'kernel_size': 3,
    'groups': 8,
    'min_value': 0.0,
    'max_value': 1.0,
    'dropout_p': 0.2,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    D = CONFIG['depth']
    H = CONFIG['height']
    W = CONFIG['width']
    K = CONFIG['kernel_size']

    x = jax.random.uniform(k1, (B, C_in, D, H, W), dtype=dtype)
    conv_weight = jax.random.normal(k2, (C_out, C_in, K, K, K), dtype=dtype) * 0.02
    conv_bias = jnp.zeros((C_out,), dtype=dtype)
    norm_weight = jnp.ones((C_out,), dtype=dtype)
    norm_bias = jnp.zeros((C_out,), dtype=dtype)
    return x, conv_weight, conv_bias, norm_weight, norm_bias


def workload(x, conv_weight, conv_bias, norm_weight, norm_bias):
    B, _, D, H, W = x.shape
    C_out = conv_weight.shape[0]
    G = CONFIG['groups']
    eps = CONFIG['eps']
    min_val = CONFIG['min_value']
    max_val = CONFIG['max_value']

    # Transpose NCDHW -> NDHWC and OIDHW -> DHWIO
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_dhwio = jnp.transpose(conv_weight, (2, 3, 4, 1, 0))

    dn = lax.conv_dimension_numbers(
        x_ndhwc.shape, w_dhwio.shape, ('NDHWC', 'DHWIO', 'NDHWC')
    )
    out_ndhwc = lax.conv_general_dilated(
        x_ndhwc,
        w_dhwio,
        window_strides=(1, 1, 1),
        padding='VALID',
        dimension_numbers=dn,
    )
    out_ndhwc = out_ndhwc + conv_bias.reshape(1, 1, 1, 1, C_out)

    # Transpose back NDHWC -> NCDHW
    x = jnp.transpose(out_ndhwc, (0, 4, 1, 2, 3))

    # GroupNorm over (C // G, D, H, W)
    _, C, D_out, H_out, W_out = x.shape
    x_grouped = x.reshape(B, G, C // G, D_out, H_out, W_out)
    mean = jnp.mean(x_grouped, axis=(2, 3, 4, 5), keepdims=True)
    var = jnp.var(x_grouped, axis=(2, 3, 4, 5), keepdims=True)
    x_norm = (x_grouped - mean) / jnp.sqrt(var + eps)
    x = x_norm.reshape(B, C, D_out, H_out, W_out)
    x = x * norm_weight.reshape(1, C, 1, 1, 1) + norm_bias.reshape(1, C, 1, 1, 1)

    # Min, Clamp, Dropout (identity in inference)
    x = jnp.minimum(x, min_val)
    x = jnp.clip(x, min_val, max_val)
    return x
