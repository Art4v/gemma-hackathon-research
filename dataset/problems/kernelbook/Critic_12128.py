"""[bf16 variant] Critic MLP: Linear -> LeakyReLU -> concat(action) -> Linear -> LeakyReLU -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Critic_12128',
    'batch': 16384,
    'state_size': 2048,
    'action_size': 1024,
    'fcs1_units': 1024,
    'fc2_units': 1024,
    'num_atoms': 128,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 5)
    state = jax.random.uniform(ks[0], (c['batch'], c['state_size']), dtype)
    action = jax.random.uniform(ks[1], (c['batch'], c['action_size']), dtype)
    w1 = jax.random.normal(ks[2], (c['state_size'], c['fcs1_units']), dtype) * 0.02
    b1 = jnp.zeros((c['fcs1_units'],), dtype)
    w2 = jax.random.normal(ks[3], (c['fcs1_units'] + c['action_size'], c['fc2_units']), dtype) * 0.02
    b2 = jnp.zeros((c['fc2_units'],), dtype)
    w3 = jax.random.normal(ks[4], (c['fc2_units'], c['num_atoms']), dtype) * 0.02
    b3 = jnp.zeros((c['num_atoms'],), dtype)
    return state, action, w1, b1, w2, b2, w3, b3


def workload(state, action, w1, b1, w2, b2, w3, b3):
    slope = CONFIG['negative_slope']
    xs = jax.nn.leaky_relu(state @ w1 + b1, negative_slope=slope)
    x = jnp.concatenate([xs, action], axis=1)
    x = jax.nn.leaky_relu(x @ w2 + b2, negative_slope=slope)
    return x @ w3 + b3
