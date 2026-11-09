"""[bf16 variant] Mean Squared Error loss: mean((predictions - targets) ** 2)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '94_MSELoss',
    'batch_size': 4096,
    'dim': 8192,
}


def create_inputs(dtype=jnp.bfloat16):
    k_pred, k_tgt, k_scale = jax.random.split(jax.random.key(0), 3)
    shape = (CONFIG['batch_size'], CONFIG['dim'])
    scale = jax.random.uniform(k_scale, (), dtype)
    predictions = jax.random.uniform(k_pred, shape, dtype) * scale
    targets = jax.random.uniform(k_tgt, shape, dtype)
    return (predictions, targets)


def workload(predictions, targets):
    diff = predictions.astype(jnp.float32) - targets.astype(jnp.float32)
    return jnp.mean(diff * diff)
