"""NAC_188: tanh(W_hat) * sigmoid(M_hat) weights followed by a bias-free linear map on the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NAC_188',
    'batch': 8,
    'height': 128,
    'width': 128,
    'n_in': 256,
    'n_out': 256,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w, k_m = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['n_in']),
        dtype,
    )
    W_hat = jax.random.normal(k_w, (CONFIG['n_in'], CONFIG['n_out']), dtype) * CONFIG['weight_scale']
    M_hat = jax.random.normal(k_m, (CONFIG['n_in'], CONFIG['n_out']), dtype) * CONFIG['weight_scale']
    return x, W_hat, M_hat


def workload(x, W_hat, M_hat):
    weights = jnp.tanh(W_hat) * jax.nn.sigmoid(M_hat)
    return x @ weights
