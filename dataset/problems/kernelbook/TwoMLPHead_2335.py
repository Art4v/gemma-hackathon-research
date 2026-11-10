"""[bf16 variant] TwoMLPHead: flatten -> Linear(fc6) -> ReLU -> Linear(fc7) -> ReLU."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TwoMLPHead_2335',
    'batch': 4096,
    'in_channels': 12544,
    'representation_size': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w6, k_b6, k_w7, k_b7 = jax.random.split(jax.random.key(0), 5)
    b = CONFIG['batch']
    c = CONFIG['in_channels']
    r = CONFIG['representation_size']
    x = jax.random.uniform(k_x, (b, c), dtype)
    w6 = jax.random.normal(k_w6, (c, r), dtype) * 0.02
    b6 = jax.random.normal(k_b6, (r,), dtype) * 0.02
    w7 = jax.random.normal(k_w7, (r, r), dtype) * 0.02
    b7 = jax.random.normal(k_b7, (r,), dtype) * 0.02
    return x, w6, b6, w7, b7


def workload(x, w6, b6, w7, b7):
    x = x.reshape(x.shape[0], -1)
    x = jax.nn.relu(x @ w6 + b6)
    x = jax.nn.relu(x @ w7 + b7)
    return x
