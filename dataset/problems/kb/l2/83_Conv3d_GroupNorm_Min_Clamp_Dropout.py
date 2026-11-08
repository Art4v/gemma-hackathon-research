"""Conv3d -> GroupNorm -> min -> clamp -> dropout (identity)."""
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
    c = CONFIG
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x, (c['batch_size'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    ks = c['kernel_size']
    w = jax.random.normal(k_w, (ks, ks, ks, c['in_channels'], c['out_channels']), dtype) * 0.02
    b = jax.random.normal(k_b, (c['out_channels'],), dtype) * 0.02
    gamma = jnp.ones((c['out_channels'],), dtype)
    beta = jnp.zeros((c['out_channels'],), dtype)
    return x, w, b, gamma, beta


def workload(x, w, b, gamma, beta):
    c = CONFIG
    xh = jnp.transpose(x, (0, 2, 3, 4, 1))  # NCDHW -> NDHWC
    y = lax.conv_general_dilated(
        xh, w, window_strides=(1, 1, 1), padding='VALID',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'))
    y = y + b
    n, d, h, wd, ch = y.shape
    g = c['groups']
    yg = y.reshape(n, d, h, wd, g, ch // g)
    mean = jnp.mean(yg, axis=(1, 2, 3, 5), keepdims=True)
    var = jnp.mean(jnp.square(yg - mean), axis=(1, 2, 3, 5), keepdims=True)
    yg = (yg - mean) * lax.rsqrt(var + c['eps'])
    y = yg.reshape(n, d, h, wd, ch) * gamma + beta
    y = jnp.minimum(y, jnp.asarray(c['min_value'], y.dtype))
    y = jnp.clip(y, c['min_value'], c['max_value'])
    # Dropout: identity in inference mode.
    return jnp.transpose(y, (0, 4, 1, 2, 3))  # NDHWC -> NCDHW
