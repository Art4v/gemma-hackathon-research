"""GatedDense: relu(Linear_h(x)) * sigmoid(Linear_g(x))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'GatedDense_15172',
    'batch': 8,
    'dim1': 32,
    'dim2': 128,
    'input_size': 1024,
    'output_size': 1024,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kwh, kbh, kwg, kbg = jax.random.split(key, 5)
    c = CONFIG
    x = jax.random.uniform(kx, (c['batch'], c['dim1'], c['dim2'], c['input_size']), dtype)
    w_h = jax.random.normal(kwh, (c['input_size'], c['output_size']), dtype) * 0.02
    b_h = jax.random.normal(kbh, (c['output_size'],), dtype) * 0.02
    w_g = jax.random.normal(kwg, (c['input_size'], c['output_size']), dtype) * 0.02
    b_g = jax.random.normal(kbg, (c['output_size'],), dtype) * 0.02
    return x, w_h, b_h, w_g, b_g


def workload(x, w_h, b_h, w_g, b_g):
    h = jax.nn.relu(x @ w_h + b_h)
    g = jax.nn.sigmoid(x @ w_g + b_g)
    return h * g
