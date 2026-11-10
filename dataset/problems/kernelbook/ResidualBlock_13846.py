"""[bf16 variant] ResidualBlock: reflect-pad -> conv3x3 -> InstanceNorm -> ReLU -> reflect-pad -> conv3x3 -> InstanceNorm -> + residual."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ResidualBlock_13846',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 1,
    'eps': 1e-9,
}


def create_inputs(dtype=jnp.bfloat16):
    n, c, h, w = CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']
    k = CONFIG['kernel_size']
    keys = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(keys[0], (n, c, h, w), dtype)
    w1 = jax.random.normal(keys[1], (k, k, c, c), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (k, k, c, c), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c,), dtype) * 0.02
    scale1 = jnp.ones((c,), dtype)
    shift1 = jnp.zeros((c,), dtype)
    scale2 = jnp.ones((c,), dtype)
    shift2 = jnp.zeros((c,), dtype)
    return x, w1, b1, scale1, shift1, w2, b2, scale2, shift2


def _conv_layer(x, w, b):
    # x: NHWC, w: HWIO. Reflection pad of floor(k/2), then VALID conv.
    p = w.shape[0] // 2
    xp = jnp.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), mode='reflect')
    s = CONFIG['stride']
    y = lax.conv_general_dilated(
        xp, w, window_strides=(s, s), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def _instance_norm(x, scale, shift):
    # Biased variance over spatial dims (torch unbiased var * (n-1)/n).
    mean = jnp.mean(x, axis=(1, 2), keepdims=True)
    var = jnp.var(x, axis=(1, 2), keepdims=True)
    out = (x - mean) / jnp.sqrt(var + CONFIG['eps'])
    return out * scale + shift


def workload(x, w1, b1, scale1, shift1, w2, b2, scale2, shift2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    out = jax.nn.relu(_instance_norm(_conv_layer(h, w1, b1), scale1, shift1))
    out = _instance_norm(_conv_layer(out, w2, b2), scale2, shift2)
    out = out + h
    return jnp.transpose(out, (0, 3, 1, 2))
