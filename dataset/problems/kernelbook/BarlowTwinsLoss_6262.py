"""Barlow Twins loss: per-feature batch standardization -> cross-correlation matmul -> on/off-diagonal squared sums."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BarlowTwinsLoss_6262',
    'batch': 16384,
    'z_dim': 2048,
    'lambda_coeff': 0.005,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    shape = (CONFIG['batch'], CONFIG['z_dim'])
    z1 = jax.random.uniform(k1, shape, dtype)
    z2 = jax.random.uniform(k2, shape, dtype)
    return z1, z2


def _standardize(z):
    # torch.std defaults to the unbiased estimator (ddof=1).
    return (z - jnp.mean(z, axis=0)) / jnp.std(z, axis=0, ddof=1)


def workload(z1, z2):
    # The module's batch_size equals the row count of its inputs.
    batch_size = z1.shape[0]
    c = jnp.matmul(_standardize(z1).T, _standardize(z2)) / batch_size
    diag = jnp.diagonal(c)
    on_diag = jnp.sum((diag - 1.0) ** 2)
    eye = jnp.eye(c.shape[0], dtype=bool)
    off_diag = jnp.sum(jnp.where(eye, 0.0, c) ** 2)
    return on_diag + CONFIG['lambda_coeff'] * off_diag
