"""BasicBlock_ins: conv3x3 -> InstanceNorm(affine) -> ReLU -> conv3x3 -> InstanceNorm(affine) -> +residual -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'BasicBlock_ins_15035',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    n, c, h, w = CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']
    k = CONFIG['kernel_size']
    kx, k1, k2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, (n, c, h, w), dtype)
    w1 = jax.random.normal(k1, (k, k, c, c), dtype) * 0.02  # HWIO
    w2 = jax.random.normal(k2, (k, k, c, c), dtype) * 0.02  # HWIO
    g1 = jnp.ones((c,), dtype)
    b1 = jnp.zeros((c,), dtype)
    g2 = jnp.ones((c,), dtype)
    b2 = jnp.zeros((c,), dtype)
    return x, w1, g1, b1, w2, g2, b2


def _conv(x, w):
    s, p = CONFIG['stride'], CONFIG['padding']
    return lax.conv_general_dilated(
        x, w, window_strides=(s, s), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))


def _instance_norm(x, gamma, beta):
    mean = jnp.mean(x, axis=(1, 2), keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=(1, 2), keepdims=True)
    return (x - mean) * lax.rsqrt(var + CONFIG['eps']) * gamma + beta


def workload(x, w1, g1, b1, w2, g2, b2):
    xh = jnp.transpose(x, (0, 2, 3, 1))
    out = jax.nn.relu(_instance_norm(_conv(xh, w1), g1, b1))
    out = _instance_norm(_conv(out, w2), g2, b2)
    out = jax.nn.relu(out + xh)
    return jnp.transpose(out, (0, 3, 1, 2))
