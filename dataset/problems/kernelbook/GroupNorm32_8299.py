"""[bf16 variant] GroupNorm (affine) followed by swish gate y * sigmoid(swish * y)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GroupNorm32_8299',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'num_groups': 1,
    'swish': 4.0,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    gamma = jnp.ones((CONFIG['channels'],), dtype)
    beta = jnp.zeros((CONFIG['channels'],), dtype)
    return x, gamma, beta


def workload(x, gamma, beta):
    x = x.astype(jnp.float32)
    n, c, h, w = x.shape
    g = CONFIG['num_groups']
    xg = x.reshape(n, g, c // g, h, w)
    mean = jnp.mean(xg, axis=(2, 3, 4), keepdims=True)
    var = jnp.mean(jnp.square(xg - mean), axis=(2, 3, 4), keepdims=True)
    y = ((xg - mean) * jax.lax.rsqrt(var + CONFIG['eps'])).reshape(n, c, h, w)
    y = y * gamma.astype(jnp.float32)[None, :, None, None] + beta.astype(jnp.float32)[None, :, None, None]
    swish = CONFIG['swish']
    if swish == 1.0:
        y = jax.nn.silu(y)
    elif swish:
        y = y * jax.nn.sigmoid(y * float(swish))
    return y
