"""Linear -> ReLU -> Linear -> ReLU -> Linear -> ReLU -> Linear (4-layer MLP over the last axis)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NN_11947',
    'batch': 32,
    'dim1': 64,
    'dim2': 64,
    'input_size': 512,
    'hidden1': 128,
    'hidden2': 64,
    'hidden3': 32,
    'num_classes': 128,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['input_size']),
        dtype,
    )
    dims = [CONFIG['input_size'], CONFIG['hidden1'], CONFIG['hidden2'],
            CONFIG['hidden3'], CONFIG['num_classes']]
    params = []
    for i in range(4):
        w = jax.random.normal(keys[1 + 2 * i], (dims[i], dims[i + 1]), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (dims[i + 1],), dtype) * 0.02
        params += [w, b]
    return (x, *params)


def workload(x, w1, b1, w2, b2, w3, b3, w4, b4):
    h = jax.nn.relu(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    h = jax.nn.relu(h @ w3 + b3)
    return h @ w4 + b4
