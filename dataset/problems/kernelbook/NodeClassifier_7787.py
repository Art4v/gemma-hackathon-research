"""NodeClassifier: Linear -> tanh -> Linear (softmax defined but unused in forward)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NodeClassifier_7787',
    'batch': 16,
    'dim1': 32,
    'dim2': 128,
    'feature_size': 1024,
    'hidden_size': 1024,
    'num_classes': 3,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    kx, k1, kb1, k2, kb2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (c['batch'], c['dim1'], c['dim2'], c['feature_size']), dtype)
    w1 = jax.random.normal(k1, (c['feature_size'], c['hidden_size']), dtype) * 0.02
    b1 = jax.random.normal(kb1, (c['hidden_size'],), dtype) * 0.02
    w2 = jax.random.normal(k2, (c['hidden_size'], c['num_classes']), dtype) * 0.02
    b2 = jax.random.normal(kb2, (c['num_classes'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jnp.tanh(x @ w1 + b1)
    return h @ w2 + b2
