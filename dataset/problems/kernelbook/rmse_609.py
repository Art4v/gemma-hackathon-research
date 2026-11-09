"""[bf16 variant] RMSE loss: mean over last axis of squared error, sqrt(+eps), then global mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'rmse_609',
    'd0': 8,
    'd1': 64,
    'd2': 128,
    'd3': 512,
    'eps': 1e-07,
}


def create_inputs(dtype=jnp.bfloat16):
    k_true, k_pred = jax.random.split(jax.random.key(0))
    shape = (CONFIG['d0'], CONFIG['d1'], CONFIG['d2'], CONFIG['d3'])
    y_true = jax.random.uniform(k_true, shape, dtype)
    y_pred = jax.random.uniform(k_pred, shape, dtype)
    return y_true, y_pred


def workload(y_true, y_pred):
    mse = jnp.mean((y_pred - y_true) ** 2, axis=-1)
    return jnp.mean(jnp.sqrt(mse + CONFIG['eps']))
