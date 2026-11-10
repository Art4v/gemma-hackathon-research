"""NeuralNet_4160: Linear -> sigmoid -> Linear over the last axis."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NeuralNet_4160',
    'batch': 8,
    'dim1': 32,
    'dim2': 128,
    'input_features': 1024,
    'hidden_layer_size': 1024,
    'output_classes': 1024,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    kx, k1, kb1, k2, kb2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(
        kx, (c['batch'], c['dim1'], c['dim2'], c['input_features']), dtype)
    w1 = jax.random.normal(k1, (c['input_features'], c['hidden_layer_size']), dtype) * 0.02
    b1 = jax.random.normal(kb1, (c['hidden_layer_size'],), dtype) * 0.02
    w2 = jax.random.normal(k2, (c['hidden_layer_size'], c['output_classes']), dtype) * 0.02
    b2 = jax.random.normal(kb2, (c['output_classes'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    h = jax.nn.sigmoid(x @ w1 + b1)
    return h @ w2 + b2
