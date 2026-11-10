"""Four-layer MLP on the last axis: Linear-ReLU-Linear-ReLU-Linear-ReLU-Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NNTest_13029',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'input_size': 256,
    'hidden1': 256,
    'hidden2': 512,
    'hidden3': 256,
    'output_size': 256,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 9)
    scale = CONFIG['weight_scale']
    x_shape = (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['input_size'])
    x = jax.random.uniform(keys[0], x_shape, dtype)
    w1 = jax.random.normal(keys[1], (CONFIG['input_size'], CONFIG['hidden1']), dtype) * scale
    b1 = jax.random.normal(keys[2], (CONFIG['hidden1'],), dtype) * scale
    w2 = jax.random.normal(keys[3], (CONFIG['hidden1'], CONFIG['hidden2']), dtype) * scale
    b2 = jax.random.normal(keys[4], (CONFIG['hidden2'],), dtype) * scale
    w3 = jax.random.normal(keys[5], (CONFIG['hidden2'], CONFIG['hidden3']), dtype) * scale
    b3 = jax.random.normal(keys[6], (CONFIG['hidden3'],), dtype) * scale
    w4 = jax.random.normal(keys[7], (CONFIG['hidden3'], CONFIG['output_size']), dtype) * scale
    b4 = jax.random.normal(keys[8], (CONFIG['output_size'],), dtype) * scale
    return x, w1, b1, w2, b2, w3, b3, w4, b4


def workload(x, w1, b1, w2, b2, w3, b3, w4, b4):
    x = jax.nn.relu(x @ w1 + b1)
    x = jax.nn.relu(x @ w2 + b2)
    x = jax.nn.relu(x @ w3 + b3)
    return x @ w4 + b4
