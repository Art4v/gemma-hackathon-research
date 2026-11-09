"""[bf16 variant] L_2nd loss: beta-weighted squared error, sum over last dim, then mean."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'L_2nd_18169',
    'shape': (8, 32, 256, 512),
    'beta': 4.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k_pred, k_true = jax.random.split(jax.random.key(0))
    y_pred = jax.random.uniform(k_pred, CONFIG['shape'], dtype)
    y_true = jax.random.uniform(k_true, CONFIG['shape'], dtype)
    return y_pred, y_true


def workload(y_pred, y_true):
    b = jnp.where(y_true != 0, jnp.asarray(CONFIG['beta'], y_true.dtype),
                  jnp.ones_like(y_true))
    x = ((y_true - y_pred) * b) ** 2
    t = jnp.sum(x, axis=-1)
    return jnp.mean(t)
