"""DDPG critic: linear + relu, concat action, linear + relu, linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DDPGCriticVersion1_182',
    'batch': 2048,
    'state_size': 4096,
    'action_size': 4096,
    'fcs1_units': 2048,
    'fc2_units': 2048,
    'output_size': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 8)
    batch = CONFIG['batch']
    state_size = CONFIG['state_size']
    action_size = CONFIG['action_size']
    fcs1_units = CONFIG['fcs1_units']
    fc2_units = CONFIG['fc2_units']
    output_size = CONFIG['output_size']
    scale = CONFIG['weight_scale']

    state = jax.random.uniform(keys[0], (batch, state_size), dtype)
    action = jax.random.uniform(keys[1], (batch, action_size), dtype)
    w1 = jax.random.normal(keys[2], (state_size, fcs1_units), dtype) * scale
    b1 = jax.random.normal(keys[3], (fcs1_units,), dtype) * scale
    w2 = jax.random.normal(keys[4], (fcs1_units + action_size, fc2_units), dtype) * scale
    b2 = jax.random.normal(keys[5], (fc2_units,), dtype) * scale
    w3 = jax.random.normal(keys[6], (fc2_units, output_size), dtype) * scale
    b3 = jax.random.normal(keys[7], (output_size,), dtype) * scale
    return state, action, w1, b1, w2, b2, w3, b3


def workload(state, action, w1, b1, w2, b2, w3, b3):
    xs = jax.nn.relu(state @ w1 + b1)
    x = jnp.concatenate((xs, action), axis=1)
    x = jax.nn.relu(x @ w2 + b2)
    return x @ w3 + b3
