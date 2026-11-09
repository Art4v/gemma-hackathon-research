"""[bf16 variant] AffineConstantFlow: z = x * exp(s) + t, log_det = sum(s, axis=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AffineConstantFlow_8329',
    'batch': 16,
    'd1': 64,
    'd2': 128,
    'dim': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, ks, kt = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, (CONFIG['batch'], CONFIG['d1'], CONFIG['d2'], CONFIG['dim']), dtype)
    s = jax.random.normal(ks, (1, CONFIG['dim']), dtype) * 0.02
    t = jax.random.normal(kt, (1, CONFIG['dim']), dtype) * 0.02
    return x, s, t


def workload(x, s, t):
    # s, t of shape (1, dim) broadcast over x's last axis, as in PyTorch.
    z = x * jnp.exp(s) + t
    log_det = jnp.sum(s, axis=1)
    return z, log_det
