"""Generator MLP: Linear -> ELU -> Linear -> sigmoid -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Generator_2161',
    'batch': 16,
    'dim1': 64,
    'dim2': 128,
    'input_size': 512,
    'hidden_size': 512,
    'output_size': 512,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    kx, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(
        kx, (c['batch'], c['dim1'], c['dim2'], c['input_size']), dtype)
    w1 = jax.random.normal(k1, (c['input_size'], c['hidden_size']), dtype) * 0.02
    b1 = jnp.zeros((c['hidden_size'],), dtype)
    w2 = jax.random.normal(k2, (c['hidden_size'], c['hidden_size']), dtype) * 0.02
    b2 = jnp.zeros((c['hidden_size'],), dtype)
    w3 = jax.random.normal(k3, (c['hidden_size'], c['output_size']), dtype) * 0.02
    b3 = jnp.zeros((c['output_size'],), dtype)
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jax.nn.elu(x @ w1 + b1)
    h = jax.nn.sigmoid(h @ w2 + b2)
    return h @ w3 + b3
