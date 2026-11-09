"""[bf16 variant] SoftmaxModel_4083: Linear -> ReLU -> Linear -> ReLU -> Linear -> Softmax(dim=1) on a 4-D input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SoftmaxModel_4083',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'num_in': 256,
    'num_hidden': 512,
    'num_out': 256,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    c = CONFIG
    x = jax.random.uniform(keys[0], (c['batch'], c['dim1'], c['dim2'], c['num_in']), dtype)
    w1 = jax.random.normal(keys[1], (c['num_in'], c['num_hidden']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['num_hidden'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (c['num_hidden'], c['num_hidden']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['num_hidden'],), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (c['num_hidden'], c['num_out']), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (c['num_out'],), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jax.nn.relu(x @ w1 + b1)
    h = jax.nn.relu(h @ w2 + b2)
    logits = h @ w3 + b3
    # nn.Softmax(dim=1) on a 4-D tensor normalizes over axis 1, not the feature axis.
    return jax.nn.softmax(logits, axis=CONFIG['softmax_axis'])
