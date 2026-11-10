"""[bf16 variant] MLP: reshape -> Linear+ReLU -> Linear+ReLU -> Linear -> log_softmax."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MLP_8911',
    'batch': 65536,
    'in_features': 784,
    'hidden1': 512,
    'hidden2': 128,
    'num_classes': 10,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    b = CONFIG['batch']
    d0, d1, d2, d3 = CONFIG['in_features'], CONFIG['hidden1'], CONFIG['hidden2'], CONFIG['num_classes']
    x = jax.random.uniform(keys[0], (b, d0), dtype)
    w1 = jax.random.normal(keys[1], (d0, d1), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (d1,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (d1, d2), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (d2,), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (d2, d3), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (d3,), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    x = x.reshape(-1, CONFIG['in_features'])
    h = jax.nn.relu(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    logits = h @ w3 + b3
    return jax.nn.log_softmax(logits, axis=1)
