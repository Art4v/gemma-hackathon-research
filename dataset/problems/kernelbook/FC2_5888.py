"""[bf16 variant] FC2: flatten -> Linear+ReLU -> Linear+ReLU -> Linear+ReLU -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'FC2_5888',
    'batch': 4096,
    'size': 128,
    'hidden1': 128,
    'hidden2': 2,
    'hidden3': 4,
    'out_features': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 9)
    b, s = CONFIG['batch'], CONFIG['size']
    dims = [s * s, CONFIG['hidden1'], CONFIG['hidden2'], CONFIG['hidden3'], CONFIG['out_features']]
    x = jax.random.uniform(keys[0], (b, s, s), dtype)
    params = []
    for i in range(4):
        w = jax.random.normal(keys[1 + 2 * i], (dims[i], dims[i + 1]), dtype) * 0.02
        bias = jax.random.normal(keys[2 + 2 * i], (dims[i + 1],), dtype) * 0.02
        params += [w, bias]
    return (x, *params)


def workload(x, w1, b1, w2, b2, w3, b3, w4, b4):
    h = x.reshape(x.shape[0], -1)
    h = jax.nn.relu(h @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    h = jax.nn.relu(h @ w3 + b3)
    return h @ w4 + b4
