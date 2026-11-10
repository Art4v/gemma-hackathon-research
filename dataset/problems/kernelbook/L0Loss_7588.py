"""L0Loss: abs difference, add eps, power gamma, mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'L0Loss_7588',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'gamma': 2,
    'eps': 1e-08,
}


def create_inputs(dtype=jnp.float32):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_pred, k_target = jax.random.split(key)
    pred = jax.random.uniform(k_pred, shape, dtype)
    target = jax.random.uniform(k_target, shape, dtype)
    return pred, target


def workload(pred, target):
    loss = jnp.power(jnp.abs(pred - target) + CONFIG['eps'], CONFIG['gamma'])
    return jnp.mean(loss)
