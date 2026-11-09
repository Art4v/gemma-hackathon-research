"""[bf16 variant] Conv3d -> multiply -> InstanceNorm3d -> clamp -> multiply -> max over channels."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max',
    'batch_size': 64,
    'in_channels': 3,
    'out_channels': 16,
    'depth': 32,
    'height': 64,
    'width': 64,
    'kernel_size': 3,
    'clamp_min': -1.0,
    'clamp_max': 1.0,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k = c['kernel_size']
    key = jax.random.key(0)
    kx, kw, kb, km = jax.random.split(key, 4)
    x = jax.random.uniform(
        kx, (c['batch_size'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['out_channels'], c['in_channels'], k, k, k), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    multiplier = jax.random.normal(km, (c['out_channels'], 1, 1, 1), dtype)
    return x, w, b, multiplier


def workload(x, w, b, multiplier):
    c = CONFIG
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    w_dhwio = jnp.transpose(w, (2, 3, 4, 1, 0))
    y = lax.conv_general_dilated(
        x_ndhwc, w_dhwio,
        window_strides=(1, 1, 1),
        padding='VALID',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    y = y + b
    m = multiplier.reshape(-1)  # (C,) broadcast over channel-last
    y = y * m
    mean = jnp.mean(y, axis=(1, 2, 3), keepdims=True)
    var = jnp.mean(jnp.square(y - mean), axis=(1, 2, 3), keepdims=True)
    y = (y - mean) * lax.rsqrt(var + c['eps'])
    y = jnp.clip(y, c['clamp_min'], c['clamp_max'])
    y = y * m
    return jnp.max(y, axis=-1)  # (N, D', H', W')
