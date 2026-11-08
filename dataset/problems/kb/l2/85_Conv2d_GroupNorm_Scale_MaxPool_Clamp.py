"""Conv2d -> GroupNorm -> channel scale -> MaxPool2d -> clamp."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '85_Conv2d_GroupNorm_Scale_MaxPool_Clamp',
    'batch_size': 128,
    'in_channels': 8,
    'out_channels': 64,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'num_groups': 16,
    'maxpool_kernel_size': 4,
    'clamp_min': 0.0,
    'clamp_max': 1.0,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(k_x, (c['batch_size'], c['in_channels'], c['height'], c['width']), dtype)
    conv_w = jax.random.normal(k_w, (c['out_channels'], c['in_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    conv_b = jax.random.normal(k_b, (c['out_channels'],), dtype) * 0.02
    gamma = jnp.ones((c['out_channels'],), dtype)
    beta = jnp.zeros((c['out_channels'],), dtype)
    scale = jnp.ones((c['out_channels'], 1, 1), dtype)
    return x, conv_w, conv_b, gamma, beta, scale


def workload(x, conv_w, conv_b, gamma, beta, scale):
    c = CONFIG
    G = c['num_groups']
    p = c['maxpool_kernel_size']
    xh = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
    y = lax.conv_general_dilated(
        xh, conv_w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
    y = y + conv_b[None, None, None, :]
    N, H, W, C = y.shape
    yg = y.reshape(N, H, W, G, C // G)
    mean = jnp.mean(yg, axis=(1, 2, 4), keepdims=True)
    var = jnp.mean(jnp.square(yg - mean), axis=(1, 2, 4), keepdims=True)
    yg = (yg - mean) * lax.rsqrt(var + c['eps'])
    y = yg.reshape(N, H, W, C)
    y = y * gamma[None, None, None, :] + beta[None, None, None, :]
    y = y * scale.reshape(1, 1, 1, C)
    y = lax.reduce_window(y, -jnp.inf, lax.max, (1, p, p, 1), (1, p, p, 1), 'VALID')
    y = jnp.clip(y, c['clamp_min'], c['clamp_max'])
    return jnp.transpose(y, (0, 3, 1, 2))  # NHWC -> NCHW
