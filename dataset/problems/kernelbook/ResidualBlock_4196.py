"""ResidualBlock: reflect-pad conv3x3 -> InstanceNorm -> ReLU -> reflect-pad conv3x3 -> InstanceNorm -> +x -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ResidualBlock_4196',
    'batch': 8,
    'filters': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'padding': 1,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    k = jax.random.split(jax.random.key(0), 5)
    n, c, h, w = CONFIG['batch'], CONFIG['filters'], CONFIG['height'], CONFIG['width']
    ks = CONFIG['kernel_size']
    x = jax.random.uniform(k[0], (n, c, h, w), dtype)
    w1 = jax.random.normal(k[1], (c, c, ks, ks), dtype) * 0.02
    b1 = jax.random.normal(k[2], (c,), dtype) * 0.02
    w2 = jax.random.normal(k[3], (c, c, ks, ks), dtype) * 0.02
    b2 = jax.random.normal(k[4], (c,), dtype) * 0.02
    g1 = jnp.ones((c,), dtype)
    be1 = jnp.zeros((c,), dtype)
    g2 = jnp.ones((c,), dtype)
    be2 = jnp.zeros((c,), dtype)
    return x, w1, b1, g1, be1, w2, b2, g2, be2


def _conv_reflect(x, w, b):
    # x: NHWC, w: OIHW (PyTorch layout).
    p = CONFIG['padding']
    xp = jnp.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), mode='reflect')
    y = lax.conv_general_dilated(
        xp, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    return y + b


def _instance_norm(x, gamma, beta):
    # x: NHWC; stats per sample and channel over H, W, biased variance.
    mean = jnp.mean(x, axis=(1, 2), keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=(1, 2), keepdims=True)
    return (x - mean) * lax.rsqrt(var + CONFIG['eps']) * gamma + beta


def workload(x, w1, b1, g1, be1, w2, b2, g2, be2):
    xh = jnp.transpose(x, (0, 2, 3, 1))
    a = jax.nn.relu(_instance_norm(_conv_reflect(xh, w1, b1), g1, be1))
    e = _instance_norm(_conv_reflect(a, w2, b2), g2, be2)
    out = jax.nn.relu(e + xh)
    return jnp.transpose(out, (0, 3, 1, 2))
