"""[bf16 variant] ConvTranspose3d -> ReLU -> GroupNorm."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '61_ConvTranspose3d_ReLU_GroupNorm',
    'batch_size': 16,
    'in_channels': 64,
    'out_channels': 128,
    'D': 32,
    'H': 32,
    'W': 32,
    'kernel_size': 3,
    'groups': 8,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(k1, (c['batch_size'], c['in_channels'], c['D'], c['H'], c['W']), dtype)
    k = c['kernel_size']
    w = jax.random.normal(k2, (c['in_channels'], c['out_channels'], k, k, k), dtype) * 0.02
    gamma = jnp.ones((c['out_channels'],), dtype)
    beta = jnp.zeros((c['out_channels'],), dtype)
    return x, w, gamma, beta


def workload(x, w, gamma, beta):
    k = w.shape[2]
    G = CONFIG['groups']
    eps = CONFIG['eps']
    xh = jnp.transpose(x, (0, 2, 3, 4, 1))  # NDHWC
    # (Cin, Cout, kD, kH, kW) -> (kD, kH, kW, Cin, Cout), spatially flipped
    wk = jnp.transpose(w, (2, 3, 4, 0, 1))[::-1, ::-1, ::-1]
    y = lax.conv_general_dilated(
        xh, wk, window_strides=(1, 1, 1),
        padding=[(k - 1, k - 1)] * 3,
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'))
    y = jnp.maximum(y, 0)
    N, D, H, W, C = y.shape
    yg = y.reshape(N, D, H, W, G, C // G)
    mean = jnp.mean(yg, axis=(1, 2, 3, 5), keepdims=True)
    var = jnp.mean(jnp.square(yg - mean), axis=(1, 2, 3, 5), keepdims=True)
    yg = (yg - mean) * lax.rsqrt(var + eps)
    y = yg.reshape(N, D, H, W, C) * gamma + beta
    return jnp.transpose(y, (0, 4, 1, 2, 3))
