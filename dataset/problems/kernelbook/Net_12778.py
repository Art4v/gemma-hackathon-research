"""Flatten -> Linear -> ReLU -> Linear -> ReLU -> Linear (three-layer MLP classifier)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Net_12778',
    'batch': 16384,
    'in_features': 2048,
    'hidden_features': 1024,
    'num_classes': 256,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    batch = CONFIG['batch']
    in_features = CONFIG['in_features']
    hidden = CONFIG['hidden_features']
    num_classes = CONFIG['num_classes']
    scale = CONFIG['weight_scale']

    keys = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(keys[0], (batch, in_features), dtype)
    w1 = jax.random.normal(keys[1], (in_features, hidden), dtype) * scale
    b1 = jax.random.normal(keys[2], (hidden,), dtype) * scale
    w2 = jax.random.normal(keys[3], (hidden, hidden), dtype) * scale
    b2 = jax.random.normal(keys[4], (hidden,), dtype) * scale
    w3 = jax.random.normal(keys[5], (hidden, num_classes), dtype) * scale
    b3 = jax.random.normal(keys[6], (num_classes,), dtype) * scale
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = x.reshape(-1, w1.shape[0])
    h = jax.nn.relu(h @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    return h @ w3 + b3
