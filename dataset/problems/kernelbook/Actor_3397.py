"""Actor_3397: Linear -> ReLU -> Linear -> ReLU -> Linear -> Tanh policy MLP."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Actor_3397',
    'batch': 8,
    'height': 128,
    'width': 128,
    'state_size': 256,
    'fc1_units': 512,
    'fc2_units': 256,
    'action_size': 128,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 7)
    scale = CONFIG['weight_scale']
    state_size = CONFIG['state_size']
    fc1_units = CONFIG['fc1_units']
    fc2_units = CONFIG['fc2_units']
    action_size = CONFIG['action_size']
    state = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], state_size),
        dtype,
    )
    w1 = jax.random.normal(keys[1], (state_size, fc1_units), dtype) * scale
    b1 = jax.random.normal(keys[2], (fc1_units,), dtype) * scale
    w2 = jax.random.normal(keys[3], (fc1_units, fc2_units), dtype) * scale
    b2 = jax.random.normal(keys[4], (fc2_units,), dtype) * scale
    w3 = jax.random.normal(keys[5], (fc2_units, action_size), dtype) * scale
    b3 = jax.random.normal(keys[6], (action_size,), dtype) * scale
    return state, w1, b1, w2, b2, w3, b3


def workload(state, w1, b1, w2, b2, w3, b3):
    x = jax.nn.relu(state @ w1 + b1)
    x = jax.nn.relu(x @ w2 + b2)
    return jnp.tanh(x @ w3 + b3)
