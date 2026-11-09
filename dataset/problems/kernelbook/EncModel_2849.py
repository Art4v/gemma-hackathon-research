"""Linear -> tanh -> Linear -> sigmoid (EncModel)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'EncModel_2849',
    'batch': 8,
    'dim1': 64,
    'dim2': 128,
    'num_input': 1024,
    'num_hid': 1024,
    'num_out': 1024,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(k_x, (c['batch'], c['dim1'], c['dim2'], c['num_input']), dtype)
    w1 = jax.random.normal(k_w1, (c['num_input'], c['num_hid']), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (c['num_hid'],), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (c['num_hid'], c['num_out']), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (c['num_out'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    hidden = jnp.tanh(x @ w1 + b1)
    return jax.nn.sigmoid(hidden @ w2 + b2)
