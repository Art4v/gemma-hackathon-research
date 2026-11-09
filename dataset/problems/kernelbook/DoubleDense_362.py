"""[bf16 variant] DoubleDense: flatten -> Linear -> ReLU -> Linear -> ReLU -> Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'DoubleDense_362',
    'batch': 8192,
    'in_channels': 4096,
    'hidden_neurons': 4096,
    'output_channels': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    b = CONFIG['batch']
    d_in = CONFIG['in_channels']
    h = CONFIG['hidden_neurons']
    h2 = h // 2
    d_out = CONFIG['output_channels']
    ks = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(ks[0], (b, d_in), dtype)
    w1 = jax.random.normal(ks[1], (d_in, h), dtype) * 0.02
    b1 = jax.random.normal(ks[2], (h,), dtype) * 0.02
    w2 = jax.random.normal(ks[3], (h, h2), dtype) * 0.02
    b2 = jax.random.normal(ks[4], (h2,), dtype) * 0.02
    w3 = jax.random.normal(ks[5], (h2, d_out), dtype) * 0.02
    b3 = jax.random.normal(ks[6], (d_out,), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def workload(x, w1, b1, w2, b2, w3, b3):
    out = x.reshape(x.shape[0], -1)
    out = jax.nn.relu(out @ w1 + b1)
    out = jax.nn.relu(out @ w2 + b2)
    return out @ w3 + b3
