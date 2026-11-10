"""[bf16 variant] QNetwork_11349: Linear -> ReLU -> Linear -> ReLU -> Linear over the last axis of a 4D state tensor."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'QNetwork_11349',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'state_size': 256,
    'fc1_units': 256,
    'fc2_units': 256,
    'action_size': 256,
    'init_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    state_shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['state_size'])
    scale = CONFIG['init_scale']
    state = jax.random.uniform(keys[0], state_shape, dtype)
    w1 = jax.random.normal(keys[1], (CONFIG['state_size'], CONFIG['fc1_units']), dtype) * scale
    b1 = jax.random.normal(keys[2], (CONFIG['fc1_units'],), dtype) * scale
    w2 = jax.random.normal(keys[3], (CONFIG['fc1_units'], CONFIG['fc2_units']), dtype) * scale
    b2 = jax.random.normal(keys[4], (CONFIG['fc2_units'],), dtype) * scale
    w3 = jax.random.normal(keys[5], (CONFIG['fc2_units'], CONFIG['action_size']), dtype) * scale
    b3 = jax.random.normal(keys[6], (CONFIG['action_size'],), dtype) * scale
    return (state, w1, b1, w2, b2, w3, b3)


def workload(state, w1, b1, w2, b2, w3, b3):
    x = jax.nn.relu(state @ w1 + b1)
    x = jax.nn.relu(x @ w2 + b2)
    return x @ w3 + b3
