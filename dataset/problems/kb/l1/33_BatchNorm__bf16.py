"""[bf16 variant] BatchNorm2d (batch statistics) over an NCHW tensor: per-channel normalize, scale, shift."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '33_BatchNorm',
    'batch_size': 8,
    'num_features': 128,
    'dim1': 256,
    'dim2': 256,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch_size'], CONFIG['num_features'], CONFIG['dim1'], CONFIG['dim2']),
        dtype,
    )
    gamma = jnp.ones((CONFIG['num_features'],), dtype)
    beta = jnp.zeros((CONFIG['num_features'],), dtype)
    return (x, gamma, beta)


def workload(x, gamma, beta):
    eps = CONFIG['eps']
    # Batch statistics over N, H, W for each channel (NCHW layout).
    mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
    centered = x - mean
    var = jnp.mean(centered * centered, axis=(0, 2, 3), keepdims=True)
    inv = jax.lax.rsqrt(var + eps)
    g = gamma.reshape(1, -1, 1, 1)
    b = beta.reshape(1, -1, 1, 1)
    return centered * inv * g + b
