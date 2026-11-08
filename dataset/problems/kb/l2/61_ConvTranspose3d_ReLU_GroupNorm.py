"""Transposed 3D convolution followed by ReLU and GroupNorm."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '61_ConvTranspose3d_ReLU_GroupNorm',
    'batch_size': 16,
    'in_channels': 128,
    'out_channels': 128,
    'D': 32,
    'H': 32,
    'W': 32,
    'kernel_size': 3,
    'groups': 8,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    B = CONFIG['batch_size']
    C_in = CONFIG['in_channels']
    C_out = CONFIG['out_channels']
    D, H, W = CONFIG['D'], CONFIG['H'], CONFIG['W']
    K = CONFIG['kernel_size']

    x = jax.random.uniform(k1, (B, C_in, D, H, W), dtype=dtype)
    weight = jax.random.normal(k2, (C_in, C_out, K, K, K), dtype=dtype) * 0.02
    gn_weight = jnp.ones((C_out,), dtype=dtype)
    gn_bias = jnp.zeros((C_out,), dtype=dtype)
    return (x, weight, gn_weight, gn_bias)


def workload(x, weight, gn_weight, gn_bias):
    B, C_in, D, H, W = x.shape
    _, C_out, kD, kH, kW = weight.shape
    groups = CONFIG['groups']
    eps = CONFIG['eps']

    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_flipped = jnp.flip(weight, axis=(2, 3, 4))
    w_dhwio = jnp.transpose(w_flipped, (2, 3, 4, 0, 1))

    padding = ((kD - 1, kD - 1), (kH - 1, kH - 1), (kW - 1, kW - 1))
    out_ndhwc = lax.conv_general_dilated(
        lhs=x_ndhwc,
        rhs=w_dhwio,
        window_strides=(1, 1, 1),
        padding=padding,
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    y = jnp.transpose(out_ndhwc, (0, 4, 1, 2, 3))

    y = jnp.maximum(y, jnp.asarray(0.0, dtype=y.dtype))

    _, _, D_out, H_out, W_out = y.shape
    y_grouped = jnp.reshape(y, (B, groups, C_out // groups, D_out, H_out, W_out))
    mean = jnp.mean(y_grouped, axis=(2, 3, 4, 5), keepdims=True)
    var = jnp.var(y_grouped, axis=(2, 3, 4, 5), keepdims=True)
    y_norm = (y_grouped - mean) / jnp.sqrt(var + eps)
    y_norm = jnp.reshape(y_norm, (B, C_out, D_out, H_out, W_out))

    y_out = y_norm * jnp.reshape(gn_weight, (1, C_out, 1, 1, 1)) + jnp.reshape(gn_bias, (1, C_out, 1, 1, 1))
    return y_out
