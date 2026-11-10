"""Element-wise add of two tensors followed by LayerNorm over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AddLayerNorm_2586',
    'batch': 8,
    'channels': 8,
    'height': 512,
    'dim': 512,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['dim'])
    key_x, key_y = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(key_x, shape, dtype)
    y = jax.random.uniform(key_y, shape, dtype)
    gamma = jnp.ones((CONFIG['dim'],), dtype)
    beta = jnp.zeros((CONFIG['dim'],), dtype)
    return x, y, gamma, beta


def workload(x, y, gamma, beta):
    s = jnp.add(x, y)
    mean = jnp.mean(s, axis=-1, keepdims=True)
    centered = s - mean
    var = jnp.mean(jnp.square(centered), axis=-1, keepdims=True)
    return centered / jnp.sqrt(var + CONFIG['eps']) * gamma + beta
