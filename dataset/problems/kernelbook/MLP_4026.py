"""MLP_4026: Linear -> ReLU -> Linear -> ReLU -> Linear -> Sigmoid applied on the last axis of a 4-D tensor."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MLP_4026',
    'batch': 8,
    'height': 128,
    'width': 128,
    'n_inputs': 256,
    'hidden1': 640,
    'hidden2': 512,
    'n_outputs': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 7)
    scale = CONFIG['weight_scale']
    n_inputs = CONFIG['n_inputs']
    hidden1 = CONFIG['hidden1']
    hidden2 = CONFIG['hidden2']
    n_outputs = CONFIG['n_outputs']

    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], n_inputs),
        dtype,
    )
    w1 = jax.random.normal(keys[1], (n_inputs, hidden1), dtype) * scale
    b1 = jax.random.normal(keys[2], (hidden1,), dtype) * scale
    w2 = jax.random.normal(keys[3], (hidden1, hidden2), dtype) * scale
    b2 = jax.random.normal(keys[4], (hidden2,), dtype) * scale
    w3 = jax.random.normal(keys[5], (hidden2, n_outputs), dtype) * scale
    b3 = jax.random.normal(keys[6], (n_outputs,), dtype) * scale
    return x, w1, b1, w2, b2, w3, b3


def workload(*inputs):
    x, w1, b1, w2, b2, w3, b3 = inputs
    h = jax.nn.relu(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    return jax.nn.sigmoid(h @ w3 + b3)
