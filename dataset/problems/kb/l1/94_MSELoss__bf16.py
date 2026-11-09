"""[bf16 variant] Mean Squared Error (MSE) loss computation."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '94_MSELoss',
    'batch_size': 8192,
    'features': 8192,
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
    diff = predictions - targets
    return jnp.mean(diff * diff)
