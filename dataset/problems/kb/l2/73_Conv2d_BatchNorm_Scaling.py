"""Conv2d -> BatchNorm2d (batch statistics) -> scale by constant factor."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '73_Conv2d_BatchNorm_Scaling',
    'batch': 128,
    'in_channels': 8,
    'out_channels': 64,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'scaling_factor': 2.0,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    c = CONFIG
    x = jax.random.uniform(kx, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['kernel_size'], c['kernel_size'], c['in_channels'], c['out_channels']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    gamma = jnp.ones((c['out_channels'],), dtype)
    beta = jnp.zeros((c['out_channels'],), dtype)
    return x, w, b, gamma, beta


def workload(x, w, b, gamma, beta):
    # x: NCHW, w: HWIO
    xh = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        xh, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    y = y + b
    mean = jnp.mean(y, axis=(0, 1, 2), keepdims=True)
    var = jnp.mean(jnp.square(y - mean), axis=(0, 1, 2), keepdims=True)
    y = (y - mean) * lax.rsqrt(var + CONFIG['eps']) * gamma + beta
    y = y * CONFIG['scaling_factor']
    return jnp.transpose(y, (0, 3, 1, 2))
