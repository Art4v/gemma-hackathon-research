"""Critic MLP: linear-relu, concat action, four linear-relu layers, linear Q-value head."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Critic_11851',
    'batch': 8192,
    'state_size': 1024,
    'action_size': 512,
    'fc1_units': 2048,
    'fc2_units': 1024,
    'fc3_units': 512,
    'fc4_units': 256,
    'fc5_units': 128,
    'output_units': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    layer_shapes = [
        (c['state_size'], c['fc1_units']),
        (c['fc1_units'] + c['action_size'], c['fc2_units']),
        (c['fc2_units'], c['fc3_units']),
        (c['fc3_units'], c['fc4_units']),
        (c['fc4_units'], c['fc5_units']),
        (c['fc5_units'], c['output_units']),
    ]
    keys = jax.random.split(jax.random.key(0), 2 + 2 * len(layer_shapes))
    state = jax.random.uniform(keys[0], (c['batch'], c['state_size']), dtype)
    action = jax.random.uniform(keys[1], (c['batch'], c['action_size']), dtype)
    params = []
    for i, (fan_in, fan_out) in enumerate(layer_shapes):
        w = jax.random.normal(keys[2 + 2 * i], (fan_in, fan_out), dtype) * c['weight_scale']
        b = jax.random.normal(keys[3 + 2 * i], (fan_out,), dtype) * c['weight_scale']
        params.extend([w, b])
    return (state, action, *params)


def workload(state, action, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6):
    x = jax.nn.relu(state @ w1 + b1)
    x = jnp.concatenate((x, action), axis=1)
    x = jax.nn.relu(x @ w2 + b2)
    x = jax.nn.relu(x @ w3 + b3)
    x = jax.nn.relu(x @ w4 + b4)
    x = jax.nn.relu(x @ w5 + b5)
    return x @ w6 + b6
