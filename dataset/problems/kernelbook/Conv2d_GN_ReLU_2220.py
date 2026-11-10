"""[bf16 variant] Conv2d (no bias, same padding) -> GroupNorm -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Conv2d_GN_ReLU_2220',
    'batch': 16,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'ksize': 3,
    'stride': 1,
    'num_groups': 1,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0))
    c = CONFIG
    x = jax.random.uniform(k1, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(k2, (c['ksize'], c['ksize'], c['in_channels'], c['out_channels']), dtype) * 0.02
    gamma = jnp.ones((c['out_channels'],), dtype)
    beta = jnp.zeros((c['out_channels'],), dtype)
    return x, w, gamma, beta


def workload(x, w, gamma, beta):
    ksize = w.shape[0]
    pad = 0 if ksize < 2 else ksize // 2
    stride = CONFIG['stride']
    g = CONFIG['num_groups']
    xh = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        xh, w, window_strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    n, h, wd, c = y.shape
    yg = y.reshape(n, h, wd, g, c // g)
    mean = jnp.mean(yg, axis=(1, 2, 4), keepdims=True)
    var = jnp.mean(jnp.square(yg - mean), axis=(1, 2, 4), keepdims=True)
    yn = ((yg - mean) * lax.rsqrt(var + CONFIG['eps'])).reshape(n, h, wd, c)
    out = jnp.maximum(yn * gamma + beta, 0)
    return jnp.transpose(out, (0, 3, 1, 2))
