"""Linear -> tanh -> Linear -> softmax(dim=1) -> weighted sum over dim 1 (additive attention pooling)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Attention_969',
    'batch': 8,
    'seq': 128,
    'inner': 128,
    'in_size': 512,
    'hidden_size': 512,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.float32):
    k = jax.random.split(jax.random.key(0), 5)
    b, n, m = CONFIG['batch'], CONFIG['seq'], CONFIG['inner']
    d, h = CONFIG['in_size'], CONFIG['hidden_size']
    x = jax.random.uniform(k[0], (b, n, m, d), dtype)
    w1 = jax.random.normal(k[1], (d, h), dtype) * 0.02
    b1 = jax.random.normal(k[2], (h,), dtype) * 0.02
    w2 = jax.random.normal(k[3], (h, d), dtype) * 0.02
    b2 = jax.random.normal(k[4], (d,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    axis = CONFIG['softmax_axis']
    hid = jnp.tanh(x @ w1 + b1)
    alpha = jax.nn.softmax(hid @ w2 + b2, axis=axis)
    return jnp.sum(alpha * x, axis=axis)
