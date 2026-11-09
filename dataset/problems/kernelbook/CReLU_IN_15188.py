"""[bf16 variant] CReLU (cat(x, -x) on channels) -> InstanceNorm2d (affine) -> LeakyReLU(0.01)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CReLU_IN_15188',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'eps': 1e-5,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']), dtype)
    gamma = jnp.ones((2 * CONFIG['channels'],), dtype)
    beta = jnp.zeros((2 * CONFIG['channels'],), dtype)
    return x, gamma, beta


def workload(x, gamma, beta):
    cat = jnp.concatenate([x, -x], axis=1)
    mean = jnp.mean(cat, axis=(2, 3), keepdims=True)
    var = jnp.mean(jnp.square(cat - mean), axis=(2, 3), keepdims=True)
    y = (cat - mean) * jax.lax.rsqrt(var + CONFIG['eps'])
    y = y * gamma[None, :, None, None] + beta[None, :, None, None]
    return jnp.where(y >= 0, y, CONFIG['negative_slope'] * y)
