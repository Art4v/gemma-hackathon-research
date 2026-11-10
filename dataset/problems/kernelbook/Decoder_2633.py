"""Decoder: Linear -> ReLU -> Linear -> Sigmoid over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Decoder_2633',
    'batch': 16,
    'dim1': 16,
    'dim2': 128,
    'latent_dim': 512,
    'hidden_dim': 1024,
    'output_dim': 512,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    c = CONFIG
    x = jax.random.uniform(
        k_x, (c['batch'], c['dim1'], c['dim2'], c['latent_dim']), dtype)
    w1 = jax.random.normal(k_w1, (c['latent_dim'], c['hidden_dim']), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (c['hidden_dim'],), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (c['hidden_dim'], c['output_dim']), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (c['output_dim'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jax.nn.relu(x @ w1 + b1)
    return jax.nn.sigmoid(h @ w2 + b2)
