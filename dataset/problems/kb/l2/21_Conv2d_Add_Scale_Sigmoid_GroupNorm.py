"""Conv2d -> add bias -> scale -> sigmoid -> GroupNorm."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '21_Conv2d_Add_Scale_Sigmoid_GroupNorm',
    'batch_size': 64,
    'in_channels': 8,
    'out_channels': 32,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'num_groups': 8,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    key = jax.random.key(0)
    kx, kw, kcb, kb, ks = jax.random.split(key, 5)
    x = jax.random.uniform(
        kx, (c['batch_size'], c['in_channels'], c['height'], c['width']), dtype)
    k = c['kernel_size']
    conv_w = jax.random.normal(
        kw, (k, k, c['in_channels'], c['out_channels']), dtype) * 0.02
    conv_b = jax.random.normal(kcb, (c['out_channels'],), dtype) * 0.02
    bias = jax.random.normal(kb, (c['out_channels'], 1, 1), dtype)
    scale = jax.random.normal(ks, (c['out_channels'], 1, 1), dtype)
    gn_gamma = jnp.ones((c['out_channels'],), dtype)
    gn_beta = jnp.zeros((c['out_channels'],), dtype)
    return x, conv_w, conv_b, bias, scale, gn_gamma, gn_beta


def workload(x, conv_w, conv_b, bias, scale, gn_gamma, gn_beta):
    G = CONFIG['num_groups']
    eps = CONFIG['eps']
    xh = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        xh, conv_w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    y = y + conv_b
    y = jnp.transpose(y, (0, 3, 1, 2))
    y = y + bias
    y = y * scale
    y = jax.nn.sigmoid(y)
    N, C, H, W = y.shape
    yg = y.reshape(N, G, C // G, H, W)
    mean = jnp.mean(yg, axis=(2, 3, 4), keepdims=True)
    var = jnp.mean(jnp.square(yg - mean), axis=(2, 3, 4), keepdims=True)
    yg = (yg - mean) * lax.rsqrt(var + eps)
    y = yg.reshape(N, C, H, W)
    return y * gn_gamma.reshape(1, C, 1, 1) + gn_beta.reshape(1, C, 1, 1)
