"""DQN: Linear -> LayerNorm -> LeakyReLU -> Linear over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DQN_3819',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'obs_size': 128,
    'hidden_size': 256,
    'num_actions': 128,
    'ln_eps': 1e-5,
    'negative_slope': 0.01,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    kx, k1, kb1, k2, kb2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (c['batch'], c['dim1'], c['dim2'], c['obs_size']), dtype)
    w1 = jax.random.normal(k1, (c['obs_size'], c['hidden_size']), dtype) * 0.02
    b1 = jax.random.normal(kb1, (c['hidden_size'],), dtype) * 0.02
    gamma = jnp.ones((c['hidden_size'],), dtype)
    beta = jnp.zeros((c['hidden_size'],), dtype)
    w2 = jax.random.normal(k2, (c['hidden_size'], c['num_actions']), dtype) * 0.02
    b2 = jax.random.normal(kb2, (c['num_actions'],), dtype) * 0.02
    return x, w1, b1, gamma, beta, w2, b2


def workload(x, w1, b1, gamma, beta, w2, b2):
    h = x @ w1 + b1
    mean = jnp.mean(h, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(h - mean), axis=-1, keepdims=True)
    h = (h - mean) / jnp.sqrt(var + CONFIG['ln_eps']) * gamma + beta
    h = jnp.where(h >= 0, h, CONFIG['negative_slope'] * h)
    return h @ w2 + b2
