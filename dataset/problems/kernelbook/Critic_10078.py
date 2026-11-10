"""[bf16 variant] Critic_10078: Linear -> LeakyReLU -> concat(action) -> Linear -> LeakyReLU -> Linear -> LeakyReLU -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Critic_10078',
    'batch': 8192,
    'state_size': 2048,
    'action_size': 2048,
    'fc1_units': 2048,
    'fc2_units': 2048,
    'fc3_units': 2048,
    'output_units': 1,
    'negative_slope': 0.01,
    'concat_axis': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    batch = CONFIG['batch']
    state_size = CONFIG['state_size']
    action_size = CONFIG['action_size']
    fc1_units = CONFIG['fc1_units']
    fc2_units = CONFIG['fc2_units']
    fc3_units = CONFIG['fc3_units']
    output_units = CONFIG['output_units']
    scale = CONFIG['weight_scale']

    keys = jax.random.split(jax.random.key(0), 10)

    state = jax.random.uniform(keys[0], (batch, state_size), dtype)
    action = jax.random.uniform(keys[1], (batch, action_size), dtype)

    w1 = jax.random.normal(keys[2], (state_size, fc1_units), dtype) * scale
    b1 = jax.random.normal(keys[3], (fc1_units,), dtype) * scale
    w2 = jax.random.normal(keys[4], (fc1_units + action_size, fc2_units), dtype) * scale
    b2 = jax.random.normal(keys[5], (fc2_units,), dtype) * scale
    w3 = jax.random.normal(keys[6], (fc2_units, fc3_units), dtype) * scale
    b3 = jax.random.normal(keys[7], (fc3_units,), dtype) * scale
    w4 = jax.random.normal(keys[8], (fc3_units, output_units), dtype) * scale
    b4 = jax.random.normal(keys[9], (output_units,), dtype) * scale

    return state, action, w1, b1, w2, b2, w3, b3, w4, b4


def workload(state, action, w1, b1, w2, b2, w3, b3, w4, b4):
    slope = CONFIG['negative_slope']
    hidden = jax.nn.leaky_relu(state @ w1 + b1, negative_slope=slope)
    hidden = jnp.concatenate((hidden, action), axis=CONFIG['concat_axis'])
    hidden = jax.nn.leaky_relu(hidden @ w2 + b2, negative_slope=slope)
    hidden = jax.nn.leaky_relu(hidden @ w3 + b3, negative_slope=slope)
    return hidden @ w4 + b4
