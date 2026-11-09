"""NTXent: L2-normalize z1, z2 -> z1 @ z2.T -> * exp(t) -> cross_entropy with arange targets (mean)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'NTXent_6903',
    'batch': 4096,
    'dim': 4096,
    'eps': 1e-12,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    b, d = CONFIG['batch'], CONFIG['dim']
    z1 = jax.random.uniform(k1, (b, d), dtype)
    z2 = jax.random.uniform(k2, (b, d), dtype)
    t = jax.random.uniform(k3, (b, b), dtype)
    return z1, z2, t


def _normalize(x):
    norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True))
    return x / jnp.maximum(norm, CONFIG['eps'])


def workload(z1, z2, t):
    z1 = _normalize(z1)
    z2 = _normalize(z2)
    sim = jnp.matmul(z1, z2.T) * jnp.exp(t)
    lse = jax.nn.logsumexp(sim, axis=-1)
    diag = jnp.diagonal(sim)
    return jnp.mean(lse - diag)
