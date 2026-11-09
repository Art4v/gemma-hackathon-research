"""Linear -> ReLU -> Linear -> softmax(dim=-1) classifier."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Classifier_6448',
    'batch': 8,
    'd1': 64,
    'd2': 128,
    'x_dim': 512,
    'h_dim': 1024,
    'y_dim': 256,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    kx, kw1, kb1, kw2, kb2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (c['batch'], c['d1'], c['d2'], c['x_dim']), dtype)
    w1 = jax.random.normal(kw1, (c['x_dim'], c['h_dim']), dtype) * 0.02
    b1 = jax.random.normal(kb1, (c['h_dim'],), dtype) * 0.02
    w2 = jax.random.normal(kw2, (c['h_dim'], c['y_dim']), dtype) * 0.02
    b2 = jax.random.normal(kb2, (c['y_dim'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jax.nn.relu(x @ w1 + b1)
    return jax.nn.softmax(h @ w2 + b2, axis=-1)
