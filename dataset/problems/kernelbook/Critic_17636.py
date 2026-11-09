"""[bf16 variant] Critic MLP: Linear -> tanh -> Linear -> tanh -> Linear(1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Critic_17636',
    'batch': 16,
    'dim1': 128,
    'dim2': 128,
    'obs_dim': 128,
    'hidden': 64,
    'out_dim': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(k[0], (c['batch'], c['dim1'], c['dim2'], c['obs_dim']), dtype)
    w1 = jax.random.normal(k[1], (c['obs_dim'], c['hidden']), dtype) * 0.02
    b1 = jax.random.normal(k[2], (c['hidden'],), dtype) * 0.02
    w2 = jax.random.normal(k[3], (c['hidden'], c['hidden']), dtype) * 0.02
    b2 = jax.random.normal(k[4], (c['hidden'],), dtype) * 0.02
    w3 = jax.random.normal(k[5], (c['hidden'], c['out_dim']), dtype) * 0.02
    b3 = jax.random.normal(k[6], (c['out_dim'],), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jnp.tanh(x @ w1 + b1)
    h = jnp.tanh(h @ w2 + b2)
    return h @ w3 + b3
