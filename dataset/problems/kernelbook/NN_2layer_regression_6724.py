"""[bf16 variant] Linear -> tanh -> Linear -> tanh -> Linear regression MLP applied on the last axis (NN_2layer_regression)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NN_2layer_regression_6724',
    'batch': 8,
    'height': 128,
    'width': 128,
    'input_dim': 512,
    'interm_dim1': 512,
    'interm_dim2': 512,
    'output_dim': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    scale = CONFIG['weight_scale']
    d = CONFIG['input_dim']
    h1 = CONFIG['interm_dim1']
    h2 = CONFIG['interm_dim2']
    out = CONFIG['output_dim']
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], d),
        dtype,
    )
    w1 = jax.random.normal(keys[1], (d, h1), dtype) * scale
    b1 = jax.random.normal(keys[2], (h1,), dtype) * scale
    w2 = jax.random.normal(keys[3], (h1, h2), dtype) * scale
    b2 = jax.random.normal(keys[4], (h2,), dtype) * scale
    w3 = jax.random.normal(keys[5], (h2, out), dtype) * scale
    b3 = jax.random.normal(keys[6], (out,), dtype) * scale
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jnp.tanh(x @ w1 + b1)
    h = jnp.tanh(h @ w2 + b2)
    return h @ w3 + b3
