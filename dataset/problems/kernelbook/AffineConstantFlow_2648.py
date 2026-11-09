"""AffineConstantFlow forward: z = x * exp(s) + t, log_det = sum(s, axis=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AffineConstantFlow_2648',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'dim': 512,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_s, k_t = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['dim']), dtype)
    s = jax.random.normal(k_s, (1, CONFIG['dim']), dtype) * 0.02
    t = jax.random.normal(k_t, (1, CONFIG['dim']), dtype) * 0.02
    return x, s, t


def workload(x, s, t):
    # s, t are (1, dim) and broadcast over the last axis of x, as in PyTorch.
    z = x * jnp.exp(s) + t
    log_det = jnp.sum(s, axis=1)
    return z, log_det
