"""[bf16 variant] MADDPG actor: Linear -> ReLU -> Linear -> ReLU -> Linear -> tanh over the last axis of a 4D state."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MADDPGActorVersion1_179',
    'dim0': 8,
    'dim1': 64,
    'dim2': 128,
    'state_size': 512,
    'fc1_units': 512,
    'fc2_units': 512,
    'action_size': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    c = CONFIG
    state = jax.random.uniform(
        keys[0], (c['dim0'], c['dim1'], c['dim2'], c['state_size']), dtype)
    w1 = jax.random.normal(keys[1], (c['state_size'], c['fc1_units']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['fc1_units'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (c['fc1_units'], c['fc2_units']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['fc2_units'],), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (c['fc2_units'], c['action_size']), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (c['action_size'],), dtype) * 0.02
    return state, w1, b1, w2, b2, w3, b3


def workload(state, w1, b1, w2, b2, w3, b3):
    x = jax.nn.relu(state @ w1 + b1)
    x = jax.nn.relu(x @ w2 + b2)
    return jnp.tanh(x @ w3 + b3)
