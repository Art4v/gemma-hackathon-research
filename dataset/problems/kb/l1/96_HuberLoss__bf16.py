"""[bf16 variant] Smooth L1 (Huber) Loss computation."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '96_HuberLoss',
    'batch_size': 8192,
    'features': 8192,
    'beta': 1.0,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    shape = (CONFIG['batch_size'], CONFIG['features'])
    scale = jax.random.uniform(k1, (), dtype=dtype)
    predictions = jax.random.uniform(k2, shape, dtype=dtype) * scale
    targets = jax.random.uniform(k3, shape, dtype=dtype)
    return (predictions, targets)

def workload(predictions, targets):
    beta = CONFIG['beta']
    diff = jnp.abs(predictions - targets)
    loss = jnp.where(diff < beta, 0.5 * (diff ** 2) / beta, diff - 0.5 * beta)
    return jnp.mean(loss)
