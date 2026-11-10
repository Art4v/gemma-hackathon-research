"""[bf16 variant] CosLoss: flatten to (-1, hidden), cosine_embedding_loss against target=1 (1 - cos), mean reduction."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CosLoss_5739',
    'batch': 8,
    'length': 128,
    'depth': 128,
    'hidden_size': 256,
    'eps': 1e-12,
}


def create_inputs(dtype=jnp.bfloat16):
    shape = (CONFIG['batch'], CONFIG['length'], CONFIG['depth'], CONFIG['hidden_size'])
    key_s, key_t = jax.random.split(jax.random.key(0))
    state_s = jax.random.uniform(key_s, shape, dtype)
    state_t = jax.random.uniform(key_t, shape, dtype)
    return state_s, state_t


def workload(state_s, state_t):
    s = state_s.reshape(-1, state_s.shape[-1])
    t = state_t.reshape(-1, state_t.shape[-1])
    eps = CONFIG['eps']
    prod_sum = jnp.sum(s * t, axis=-1)
    mag_square_s = jnp.sum(s * s, axis=-1) + eps
    mag_square_t = jnp.sum(t * t, axis=-1) + eps
    cos = prod_sum / jnp.sqrt(mag_square_s * mag_square_t)
    return jnp.mean(1.0 - cos)
