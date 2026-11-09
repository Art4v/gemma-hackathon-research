"""AdaFM: tile per-(in,out) gamma/beta over HxH, then out = gamma * x + beta."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AdaFM_8536',
    'batch': 16,
    'channels': 256,
    'height': 128,
    'width': 128,
    # PyTorch broadcasting lines in_channel up with the batch dim and out_channel with channels.
    'in_channel': 16,
    'out_channel': 256,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    pshape = (CONFIG['in_channel'], CONFIG['out_channel'], 1, 1)
    gamma = jnp.ones(pshape, dtype)
    beta = jnp.zeros(pshape, dtype)
    return x, gamma, beta


def workload(x, gamma, beta):
    h = x.shape[2]
    g = jnp.tile(gamma, (1, 1, h, h))
    b = jnp.tile(beta, (1, 1, h, h))
    return g * x + b
