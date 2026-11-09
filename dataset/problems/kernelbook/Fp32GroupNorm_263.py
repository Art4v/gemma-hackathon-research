"""Fp32GroupNorm: upcast to f32, GroupNorm(num_groups) with affine weight/bias, cast back to input dtype."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Fp32GroupNorm_263',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'num_groups': 1,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    weight = jnp.ones((CONFIG['channels'],), dtype)
    bias = jnp.zeros((CONFIG['channels'],), dtype)
    return x, weight, bias


def workload(x, weight, bias):
    n, c, h, w = x.shape
    g = CONFIG['num_groups']
    xf = x.astype(jnp.float32).reshape(n, g, c // g, h, w)
    mean = jnp.mean(xf, axis=(2, 3, 4), keepdims=True)
    var = jnp.mean(jnp.square(xf - mean), axis=(2, 3, 4), keepdims=True)
    y = ((xf - mean) * jax.lax.rsqrt(var + CONFIG['eps'])).reshape(n, c, h, w)
    y = y * weight.astype(jnp.float32)[None, :, None, None] + bias.astype(jnp.float32)[None, :, None, None]
    return y.astype(x.dtype)
