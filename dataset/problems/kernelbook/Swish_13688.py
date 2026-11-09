"""[bf16 variant] Swish with per-feature learnable beta: x * sigmoid(betas * x) over the last dim."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Swish_13688',
    'batch': 16,
    'channels': 64,
    'height': 128,
    'dim': 256,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['dim'])
    x = jax.random.uniform(key, shape, dtype)
    betas = jnp.ones((CONFIG['dim'],), dtype)
    return x, betas


def workload(x, betas):
    dim = betas.shape[0]
    gate = jax.nn.sigmoid(betas.reshape(-1, dim) * x.reshape(-1, dim))
    return x * gate.reshape(x.shape)
