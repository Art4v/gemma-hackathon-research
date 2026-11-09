"""Features_2_to_1: diagonal / diag-max / row-max / col-max / global-max stacked into (b, n, basis*f)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Features_2_to_1_831',
    'batch': 8,
    'n_vertices': 256,
    'in_features': 128,
    'basis': 5,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['n_vertices'], CONFIG['n_vertices'], CONFIG['in_features'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    b, n, _, f = x.shape
    diag_part = jnp.transpose(jnp.diagonal(x, axis1=1, axis2=2), (0, 2, 1))  # (b, n, f)
    max_diag = jnp.max(diag_part, axis=1, keepdims=True)
    max_rows = jnp.max(x, axis=2)
    max_cols = jnp.max(x, axis=1)
    max_all = jnp.max(x, axis=(1, 2), keepdims=True)[:, 0]  # (b, 1, f)
    ops = [
        diag_part,
        jnp.broadcast_to(max_diag, diag_part.shape),
        max_rows,
        max_cols,
        jnp.broadcast_to(max_all, diag_part.shape),
    ]
    out = jnp.stack(ops, axis=2)
    return out.reshape(b, n, CONFIG['basis'] * f)
