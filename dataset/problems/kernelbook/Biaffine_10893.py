"""Biaffine: (x_l @ M) @ x_r^T + bias + linear_l(x_l) + linear_r(x_r)^T."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Biaffine_10893',
    'batch': 16,
    'len_left': 1024,
    'len_right': 1024,
    'dim_left': 1024,
    'dim_right': 1024,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 8)
    x_l = jax.random.uniform(ks[0], (c['batch'], c['len_left'], c['dim_left']), dtype)
    x_r = jax.random.uniform(ks[1], (c['batch'], c['len_right'], c['dim_right']), dtype)
    matrix = jax.random.normal(ks[2], (c['dim_left'], c['dim_right']), dtype) * 0.02
    bias = jax.random.normal(ks[3], (1,), dtype) * 0.02
    w_l = jax.random.normal(ks[4], (c['dim_left'], 1), dtype) * 0.02
    b_l = jax.random.normal(ks[5], (1,), dtype) * 0.02
    w_r = jax.random.normal(ks[6], (c['dim_right'], 1), dtype) * 0.02
    b_r = jax.random.normal(ks[7], (1,), dtype) * 0.02
    return x_l, x_r, matrix, bias, w_l, b_l, w_r, b_r


def workload(x_l, x_r, matrix, bias, w_l, b_l, w_r, b_r):
    x = jnp.einsum('bld,brd->blr', x_l @ matrix, x_r) + bias
    left = x_l @ w_l + b_l                                   # (B, L, 1)
    right = jnp.swapaxes(x_r @ w_r + b_r, 1, 2)              # (B, 1, R)
    return x + left + right
