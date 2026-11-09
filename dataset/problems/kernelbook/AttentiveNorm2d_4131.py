"""[bf16 variant] AttentiveNorm2d: BatchNorm2d(batch stats, no affine) -> avgpool -> Linear -> sigmoid -> mixture gamma/beta affine."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'AttentiveNorm2d_4131',
    'batch': 16,
    'num_features': 128,
    'height': 128,
    'width': 128,
    'hidden_channels': 32,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    n, c = CONFIG['batch'], CONFIG['num_features']
    h, w, k = CONFIG['height'], CONFIG['width'], CONFIG['hidden_channels']
    keys = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(keys[0], (n, c, h, w), dtype)
    fc_w = jax.random.normal(keys[1], (c, k), dtype) * 0.02
    fc_b = jax.random.normal(keys[2], (k,), dtype) * 0.02
    gamma = jax.random.normal(keys[3], (k, c), dtype) * 0.02
    beta = jax.random.normal(keys[4], (k, c), dtype) * 0.02
    return x, fc_w, fc_b, gamma, beta


def workload(x, fc_w, fc_b, gamma, beta):
    hp = lax.Precision.HIGHEST
    mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=(0, 2, 3), keepdims=True)
    xhat = (x - mean) * lax.rsqrt(var + CONFIG['eps'])
    y = jnp.mean(x, axis=(2, 3))
    y = jax.nn.sigmoid(jnp.matmul(y, fc_w, precision=hp) + fc_b)
    g = jnp.matmul(y, gamma, precision=hp)
    b = jnp.matmul(y, beta, precision=hp)
    return g[:, :, None, None] * xhat + b[:, :, None, None]
