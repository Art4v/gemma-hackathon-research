"""[bf16 variant] Hinge loss: mean(clamp(1 - predictions * targets, min=0))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '100_HingeLoss',
    'batch_size': 4096,
    'dim': 16384,
    'clamp_min': 0.0,
    'margin': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    predictions = jax.random.uniform(k1, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    # targets broadcast against the last axis of predictions (PyTorch semantics)
    targets = (jax.random.randint(k2, (CONFIG['dim'],), 0, 2).astype(dtype) * 2 - 1)
    return predictions, targets


def workload(predictions, targets):
    hinge = jnp.maximum(CONFIG['margin'] - predictions * targets, CONFIG['clamp_min'])
    return jnp.mean(hinge)
