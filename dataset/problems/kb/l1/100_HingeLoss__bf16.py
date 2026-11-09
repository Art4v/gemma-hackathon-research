"""[bf16 variant] Hinge Loss for binary classification: mean(maximum(0, 1 - predictions * targets))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '100_HingeLoss',
    'batch_size': 8192,
    'features': 8192,
    'margin': 1.0,
}

def create_inputs(dtype=jnp.bfloat16):
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    B = CONFIG['batch_size']
    F = CONFIG['features']
    predictions = jax.random.uniform(k1, (B, F), dtype=dtype)
    targets_int = jax.random.randint(k2, (F,), 0, 2)
    targets = (targets_int.astype(dtype) * 2.0) - 1.0
    return (predictions, targets)

def workload(predictions, targets):
    margin = CONFIG['margin']
    return jnp.mean(jnp.maximum(0.0, margin - predictions * targets))
