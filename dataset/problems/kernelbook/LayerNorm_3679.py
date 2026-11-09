"""[bf16 variant] LayerNorm over channel axis 1 (mean/var/rsqrt) with gamma/beta broadcast as view(1,-1,1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'LayerNorm_3679',
    'batch': 8,
    'channels': 256,
    'height': 256,
    'width': 128,
    'eps': 1e-5,
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
    # Mirrors torch gamma.view(1, -1, 1): right-aligned broadcast hits axis 2 (H), not axis 1.
    return y * gamma.reshape(1, -1, 1) + beta.reshape(1, -1, 1)
