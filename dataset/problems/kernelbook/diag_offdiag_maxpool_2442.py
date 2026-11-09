"""[bf16 variant] diag_offdiag_maxpool: diagonal max, global-range shift, full-matrix max, concat."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'diag_offdiag_maxpool_2442',
    'batch': 16,
    'channels': 128,
    'n': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch'], CONFIG['channels'], CONFIG['n'], CONFIG['n']), dtype)
    return (x,)


def workload(x):
    max_diag = jnp.max(jnp.diagonal(x, axis1=-2, axis2=-1), axis=2)
    max_val = jnp.max(max_diag)
    min_val = jnp.max(-x)
    val = jnp.abs(max_val + min_val)
    # The torch diag_embed(x[0][0]) * 0 + val, then diagonal(), yields an all-val (N, N) matrix.
    max_offdiag = jnp.max(x - val, axis=(2, 3))
    return jnp.concatenate((max_diag, max_offdiag), axis=1)
