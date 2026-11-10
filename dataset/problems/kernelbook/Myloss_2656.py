"""[bf16 variant] Myloss: weighted binary cross-entropy (log, multiply, subtract, sum, divide by 2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Myloss_2656',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'width': 128,
    'epsilon': 1e-08,
    'divisor': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    key = jax.random.key(0)
    k_input, k_label, k_weight = jax.random.split(key, 3)
    input_ = jax.random.uniform(k_input, shape, dtype)
    label = jax.random.uniform(k_label, shape, dtype)
    weight = jax.random.uniform(k_weight, shape, dtype)
    return input_, label, weight


def workload(input_, label, weight):
    epsilon = CONFIG['epsilon']
    entropy = -label * jnp.log(input_ + epsilon) - (1 - label) * jnp.log(1 - input_ + epsilon)
    return jnp.sum(entropy * weight) / CONFIG['divisor']
