"""LayerNormalization_18037: last-axis mean/std (ddof=1) normalization with eps, then gamma scale and beta shift."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerNormalization_18037',
    'batch': 8,
    'dim1': 128,
    'dim2': 128,
    'd_hid': 256,
    'eps': 0.001,
    'ddof': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    z = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['d_hid']),
        dtype,
    )
    gamma = jnp.ones((CONFIG['d_hid'],), dtype)
    beta = jnp.zeros((CONFIG['d_hid'],), dtype)
    return z, gamma, beta


def workload(z, gamma, beta):
    mean = jnp.mean(z, axis=-1, keepdims=True)
    std = jnp.std(z, axis=-1, keepdims=True, ddof=CONFIG['ddof'])
    ln_out = (z - mean) / (std + CONFIG['eps'])
    return gamma * ln_out + beta
