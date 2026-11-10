"""[bf16 variant] Channel LayerNorm: mean/var over dim 1, rsqrt normalize, per-channel affine (gamma, beta)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerNorm_9702',
    'batch': 16,
    'channels': 256,
    'height': 128,
    'width': 128,
    'eps': 1e-4,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    gamma = jnp.ones((CONFIG['channels'],), dtype)
    beta = jnp.zeros((CONFIG['channels'],), dtype)
    return x, gamma, beta


def workload(x, gamma, beta):
    mean = jnp.mean(x, axis=1, keepdims=True)
    variance = jnp.mean((x - mean) ** 2, axis=1, keepdims=True)
    y = (x - mean) * jax.lax.rsqrt(variance + CONFIG['eps'])
    shape = (1, -1) + (1,) * (x.ndim - 2)
    return y * gamma.reshape(shape) + beta.reshape(shape)
