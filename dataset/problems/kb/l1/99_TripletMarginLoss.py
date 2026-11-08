"""Triplet margin loss with L2 pairwise distance."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '99_TripletMarginLoss',
    'batch_size': 8192,
    'features': 4096,
    'margin': 1.0,
    'eps': 1e-6,
}

def create_inputs(dtype=jnp.float32):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    shape = (CONFIG['batch_size'], CONFIG['features'])
    scale = jax.random.uniform(k1, (), dtype=dtype)
    anchor = jax.random.uniform(k2, shape, dtype=dtype) * scale
    positive = jax.random.uniform(k3, shape, dtype=dtype)
    negative = jax.random.uniform(k4, shape, dtype=dtype)
    return anchor, positive, negative

def workload(anchor, positive, negative):
    eps = CONFIG['eps']
    margin = CONFIG['margin']
    d_pos = jnp.sqrt(jnp.sum(jnp.square(anchor - positive + eps), axis=-1))
    d_neg = jnp.sqrt(jnp.sum(jnp.square(anchor - negative + eps), axis=-1))
    losses = jnp.maximum(d_pos - d_neg + margin, 0.0)
    return jnp.mean(losses)
