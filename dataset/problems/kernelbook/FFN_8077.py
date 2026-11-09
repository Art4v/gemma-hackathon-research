"""[bf16 variant] FFN: Linear -> ReLU -> Linear -> elementwise multiply by mask."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FFN_8077',
    'batch': 8,
    'dim1': 64,
    'dim2': 64,
    'd_model': 1024,
    'hidden_size': 4096,
}


def create_inputs(dtype=jnp.bfloat16):
    k = jax.random.split(jax.random.key(0), 6)
    shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['d_model'])
    d, h = CONFIG['d_model'], CONFIG['hidden_size']
    x = jax.random.uniform(k[0], shape, dtype)
    x_mas = jax.random.uniform(k[1], shape, dtype)
    w1 = jax.random.normal(k[2], (d, h), dtype) * 0.02
    b1 = jax.random.normal(k[3], (h,), dtype) * 0.02
    w2 = jax.random.normal(k[4], (h, d), dtype) * 0.02
    b2 = jax.random.normal(k[5], (d,), dtype) * 0.02
    return x, x_mas, w1, b1, w2, b2


def workload(x, x_mas, w1, b1, w2, b2):
    hidden = jax.nn.relu(x @ w1 + b1)
    return (hidden @ w2 + b2) * x_mas
