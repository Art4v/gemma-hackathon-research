"""[bf16 variant] AffineChannel2d: per-channel scale and shift, y = x * w[c] + b[c] on NCHW input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AffineChannel2d_13241',
    'batch': 8,
    'num_features': 128,
    'height': 256,
    'width': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw = jax.random.split(jax.random.key(0))
    c = CONFIG['num_features']
    x = jax.random.uniform(
        kx, (CONFIG['batch'], c, CONFIG['height'], CONFIG['width']), dtype)
    # Matches the torch init: weight.uniform_(), bias.zero_().
    weight = jax.random.uniform(kw, (c,), dtype)
    bias = jnp.zeros((c,), dtype)
    return x, weight, bias


def workload(x, weight, bias):
    c = weight.shape[0]
    return x * weight.reshape(1, c, 1, 1) + bias.reshape(1, c, 1, 1)
