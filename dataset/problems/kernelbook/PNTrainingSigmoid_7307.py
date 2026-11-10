"""[bf16 variant] PNTrainingSigmoid: prior * mean(sigmoid(-output_p)) + (1 - prior) * mean(sigmoid(output_n))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'PNTrainingSigmoid_7307',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key_p, key_n, key_prior = jax.random.split(jax.random.key(0), 3)
    output_p = jax.random.uniform(key_p, shape, dtype)
    output_n = jax.random.uniform(key_n, shape, dtype)
    prior = jax.random.uniform(key_prior, shape, dtype)
    return output_p, output_n, prior


def workload(output_p, output_n, prior):
    cost = prior * jnp.mean(jax.nn.sigmoid(-output_p))
    cost = cost + (1 - prior) * jnp.mean(jax.nn.sigmoid(output_n))
    return cost
