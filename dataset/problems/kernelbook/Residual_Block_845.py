"""[bf16 variant] Residual block: Conv3x3 -> InstanceNorm(affine) -> LeakyReLU(0.2) -> Conv3x3 -> InstanceNorm(affine) -> + identity."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Residual_Block_845',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'eps': 1e-5,
    'negative_slope': 0.2,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w1, k_w2 = jax.random.split(jax.random.key(0), 3)
    c, ks = CONFIG['channels'], CONFIG['kernel_size']
    x = jax.random.uniform(k_x, (CONFIG['batch'], c, CONFIG['height'], CONFIG['width']), dtype)
    w1 = jax.random.normal(k_w1, (ks, ks, c, c), dtype) * 0.02  # HWIO
    w2 = jax.random.normal(k_w2, (ks, ks, c, c), dtype) * 0.02  # HWIO
    g1 = jnp.ones((c,), dtype)
    b1 = jnp.zeros((c,), dtype)
    g2 = jnp.ones((c,), dtype)
    b2 = jnp.zeros((c,), dtype)
    return x, w1, w2, g1, b1, g2, b2


def _conv(x, w):
    return lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding='SAME',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))


def _instance_norm(x, gamma, beta):
    mean = jnp.mean(x, axis=(1, 2), keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=(1, 2), keepdims=True)
    return (x - mean) * lax.rsqrt(var + CONFIG['eps']) * gamma + beta


def workload(x, w1, w2, g1, b1, g2, b2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    out = _instance_norm(_conv(h, w1), g1, b1)
    out = jnp.where(out >= 0, out, out * CONFIG['negative_slope'])
    out = _instance_norm(_conv(out, w2), g2, b2)
    out = out + h
    return jnp.transpose(out, (0, 3, 1, 2))
