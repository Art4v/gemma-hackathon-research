"""[bf16 variant] Highway: sigmoid(fc1(x)) * relu(fc2(x)) + (1 - sigmoid(fc1(x))) * x."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Highway_8386',
    'dim0': 8,
    'dim1': 8,
    'dim2': 256,
    'input_size': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    k = jax.random.split(jax.random.key(0), 5)
    d = CONFIG['input_size']
    x = jax.random.uniform(k[0], (CONFIG['dim0'], CONFIG['dim1'], CONFIG['dim2'], d), dtype)
    w1 = jax.random.normal(k[1], (d, d), dtype) * 0.02
    b1 = jax.random.normal(k[2], (d,), dtype) * 0.02
    w2 = jax.random.normal(k[3], (d, d), dtype) * 0.02
    b2 = jax.random.normal(k[4], (d,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    t = jax.nn.sigmoid(x @ w1 + b1)
    h = jax.nn.relu(x @ w2 + b2)
    return t * h + (1.0 - t) * x
