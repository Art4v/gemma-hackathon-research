"""[bf16 variant] AutoEncoder: Linear(Q->k) -> sigmoid -> Linear(k->Q) -> sigmoid."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AutoEncoder_3580',
    'batch_shape': (8, 32, 128),
    'num_question': 1024,
    'k': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    q, k = CONFIG['num_question'], CONFIG['k']
    kx, kg, kgb, kh, khb = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (*CONFIG['batch_shape'], q), dtype)
    w_g = jax.random.normal(kg, (q, k), dtype) * 0.02
    b_g = jax.random.normal(kgb, (k,), dtype) * 0.02
    w_h = jax.random.normal(kh, (k, q), dtype) * 0.02
    b_h = jax.random.normal(khb, (q,), dtype) * 0.02
    return x, w_g, b_g, w_h, b_h


def workload(x, w_g, b_g, w_h, b_h):
    hidden = jax.nn.sigmoid(x @ w_g + b_g)
    return jax.nn.sigmoid(hidden @ w_h + b_h)
