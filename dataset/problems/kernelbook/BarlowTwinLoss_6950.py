"""[bf16 variant] Barlow Twins loss: row L2-normalize, cross-correlation matmul, on/off-diagonal squared sums."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BarlowTwinLoss_6950',
    'batch': 16384,
    'dim': 1024,
    'lambda_param': 0.005,
    'eps': 1e-12,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, ky = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['dim'])
    x = jax.random.uniform(kx, shape, dtype)
    y = jax.random.uniform(ky, shape, dtype)
    return x, y


def _normalize(v):
    norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.maximum(norm, CONFIG['eps'])


def workload(x, y):
    n, d = x.shape
    c = jnp.matmul(_normalize(x).T, _normalize(y)) / n
    on_diag = jnp.sum((jnp.diagonal(c) - 1.0) ** 2)
    off_mask = ~jnp.eye(d, dtype=bool)
    off_diag = jnp.sum(jnp.where(off_mask, c * c, 0.0))
    return on_diag + CONFIG['lambda_param'] * off_diag
