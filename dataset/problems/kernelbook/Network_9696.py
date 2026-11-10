"""Linear -> ReLU -> Linear over the last axis of a 4-D tensor (Network_9696)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Network_9696',
    'batch': 8,
    'channels': 128,
    'height': 128,
    'in_features': 128,
    'hidden_features': 768,
    'out_features': 128,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['in_features']),
        dtype,
    )
    w1 = jax.random.normal(keys[1], (CONFIG['in_features'], CONFIG['hidden_features']), dtype) * CONFIG['weight_scale']
    b1 = jax.random.normal(keys[2], (CONFIG['hidden_features'],), dtype) * CONFIG['weight_scale']
    w2 = jax.random.normal(keys[3], (CONFIG['hidden_features'], CONFIG['out_features']), dtype) * CONFIG['weight_scale']
    b2 = jax.random.normal(keys[4], (CONFIG['out_features'],), dtype) * CONFIG['weight_scale']
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jax.nn.relu(x @ w1 + b1)
    return h @ w2 + b2
