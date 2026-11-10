"""Critic MLP: Linear -> tanh -> Linear -> tanh -> Linear -> squeeze."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Critic_4231',
    'batch': 16,
    'dim1': 128,
    'dim2': 128,
    'obs_dim': 256,
    'hidden': 64,
    'out_dim': 1,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(ks[0], (c['batch'], c['dim1'], c['dim2'], c['obs_dim']), dtype)
    w1 = jax.random.normal(ks[1], (c['obs_dim'], c['hidden']), dtype) * 0.02
    b1 = jax.random.normal(ks[2], (c['hidden'],), dtype) * 0.02
    w2 = jax.random.normal(ks[3], (c['hidden'], c['hidden']), dtype) * 0.02
    b2 = jax.random.normal(ks[4], (c['hidden'],), dtype) * 0.02
    w3 = jax.random.normal(ks[5], (c['hidden'], c['out_dim']), dtype) * 0.02
    b3 = jax.random.normal(ks[6], (c['out_dim'],), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jnp.tanh(x @ w1 + b1)
    h = jnp.tanh(h @ w2 + b2)
    return jnp.squeeze(h @ w3 + b3)
