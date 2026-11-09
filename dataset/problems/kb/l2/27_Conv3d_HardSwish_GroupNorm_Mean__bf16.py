"""[bf16 variant] Conv3d -> HardSwish -> GroupNorm -> spatial mean."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '27_Conv3d_HardSwish_GroupNorm_Mean',
    'batch': 1024,
    'in_channels': 3,
    'out_channels': 16,
    'depth': 16,
    'height': 32,
    'width': 32,
    'kernel_size': 4,
    'num_groups': 4,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    x = jax.random.uniform(
        kx, (c['batch'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    k = c['kernel_size']
    w = jax.random.normal(kw, (k, k, k, c['in_channels'], c['out_channels']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    gamma = jnp.ones((c['out_channels'],), dtype)
    beta = jnp.zeros((c['out_channels'],), dtype)
    return x, w, b, gamma, beta


def workload(x, w, b, gamma, beta):
    G = CONFIG['num_groups']
    eps = CONFIG['eps']
    xh = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
    y = lax.conv_general_dilated(
        xh, w, window_strides=(1, 1, 1), padding='VALID',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        precision=lax.Precision.HIGHEST)
    y = y + b
    y = y * jnp.clip(y + 3.0, 0.0, 6.0) / 6.0
    B, D, H, W, C = y.shape
    yg = y.reshape(B, D, H, W, G, C // G)
    mean = jnp.mean(yg, axis=(1, 2, 3, 5), keepdims=True)
    var = jnp.mean(jnp.square(yg - mean), axis=(1, 2, 3, 5), keepdims=True)
    yn = ((yg - mean) * lax.rsqrt(var + eps)).reshape(B, D, H, W, C)
    yn = yn * gamma + beta
    return jnp.mean(yn, axis=(1, 2, 3))
