"""[bf16 variant] ResnetBlockFC: relu -> linear -> relu -> linear -> identity residual add."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ResnetBlockFC_2418',
    'batch': 8,
    'seq': 4096,
    'size_in': 1024,
    'size_h': 1024,
    'size_out': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, k0, k1 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, (CONFIG['batch'], CONFIG['seq'], CONFIG['size_in']), dtype)
    w0 = jax.random.normal(k0, (CONFIG['size_in'], CONFIG['size_h']), dtype) * 0.02
    b0 = jnp.zeros((CONFIG['size_h'],), dtype)
    w1 = jax.random.normal(k1, (CONFIG['size_h'], CONFIG['size_out']), dtype) * 0.02
    b1 = jnp.zeros((CONFIG['size_out'],), dtype)
    return x, w0, b0, w1, b1


def workload(x, w0, b0, w1, b1):
    # size_in == size_out, so the PyTorch shortcut is None and the residual is x.
    net = jax.nn.relu(x) @ w0 + b0
    dx = jax.nn.relu(net) @ w1 + b1
    return x + dx
