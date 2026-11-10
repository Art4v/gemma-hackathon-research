"""[bf16 variant] GCT (l2 mode): spatial L2 embedding -> channel-normalized gate -> 1 + tanh -> x * gate."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GCT_6827',
    'batch': 16,
    'channels': 256,
    'height': 128,
    'width': 128,
    'epsilon': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    b, c, h, w = CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']
    x = jax.random.uniform(key, (b, c, h, w), dtype)
    alpha = jnp.ones((1, c, 1, 1), dtype)
    gamma = jnp.zeros((1, c, 1, 1), dtype)
    beta = jnp.zeros((1, c, 1, 1), dtype)
    return x, alpha, gamma, beta


def workload(x, alpha, gamma, beta):
    eps = CONFIG['epsilon']
    embedding = jnp.sqrt(jnp.sum(x * x, axis=(2, 3), keepdims=True) + eps) * alpha
    norm = gamma / jnp.sqrt(jnp.mean(embedding * embedding, axis=1, keepdims=True) + eps)
    gate = 1.0 + jnp.tanh(embedding * norm + beta)
    return x * gate
