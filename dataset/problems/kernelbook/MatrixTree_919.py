"""[bf16 variant] MatrixTree: exp -> masked Laplacian -> batched matrix inverse -> matrix-tree marginals."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MatrixTree_919',
    'batch': 32,
    'n': 1024,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch'], CONFIG['n'], CONFIG['n']), dtype)
    return (x,)


def workload(x):
    n = x.shape[-1]
    eps = CONFIG['eps']
    row = lax.broadcasted_iota(jnp.int32, (n, n), 0)
    col = lax.broadcasted_iota(jnp.int32, (n, n), 1)
    is_diag = row == col

    ex = jnp.exp(x)
    diag_exp = jnp.diagonal(ex, axis1=-2, axis2=-1)  # (B, N)

    lap = jnp.where(is_diag, 0.0, ex + eps).astype(x.dtype)
    col_sums = lap.sum(axis=-2)  # torch lap.sum(0) per batch
    lap = -lap + jnp.where(is_diag, col_sums[:, None, :], 0.0)
    lap = jnp.where(row == 0, diag_exp[:, None, :], lap)

    inv = jnp.linalg.inv(lap)
    inv_t = jnp.swapaxes(inv, -1, -2)
    inv_diag = jnp.diagonal(inv, axis1=-2, axis2=-1)  # (B, N)

    term1 = jnp.where(col == 0, 0.0, ex * inv_diag[:, None, :])
    term2 = jnp.where(row == 0, 0.0, ex * inv_t)
    roots = diag_exp * inv_t[:, 0, :]  # (B, N)
    out = term1 - term2 + jnp.where(is_diag, roots[:, None, :], 0.0)
    return out.astype(x.dtype)
