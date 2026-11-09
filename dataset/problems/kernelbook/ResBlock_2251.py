"""[bf16 variant] ResBlock: zero-pad -> conv3x3 -> InstanceNorm -> LeakyReLU(0.2) -> zero-pad -> conv3x3 -> InstanceNorm -> + residual."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ResBlock_2251',
    'batch': 8,
    'dim': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'padding': 1,
    'eps': 1e-5,
    'negative_slope': 0.2,
}


def create_inputs(dtype=jnp.bfloat16):
    k = jax.random.split(jax.random.key(0), 5)
    n, c, h, w = CONFIG['batch'], CONFIG['dim'], CONFIG['height'], CONFIG['width']
    ks = CONFIG['kernel_size']
    x = jax.random.uniform(k[0], (n, c, h, w), dtype)
    w1 = jax.random.normal(k[1], (ks, ks, c, c), dtype) * 0.02
    b1 = jax.random.normal(k[2], (c,), dtype) * 0.02
    w2 = jax.random.normal(k[3], (ks, ks, c, c), dtype) * 0.02
    b2 = jax.random.normal(k[4], (c,), dtype) * 0.02
    return x, w1, b1, w2, b2


def _conv(x, w, b):
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    return y + b


def _instance_norm(x):
    mean = jnp.mean(x, axis=(1, 2), keepdims=True)
    var = jnp.var(x, axis=(1, 2), keepdims=True)
    return (x - mean) * lax.rsqrt(var + CONFIG['eps'])


def workload(x, w1, b1, w2, b2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    y = _instance_norm(_conv(h, w1, b1))
    y = jnp.where(y >= 0, y, CONFIG['negative_slope'] * y)
    y = _instance_norm(_conv(y, w2, b2))
    return jnp.transpose(y, (0, 3, 1, 2)) + x
