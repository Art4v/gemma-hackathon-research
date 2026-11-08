"""Upper triangular matrix multiplication: C = triu(A @ B)."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': '14_Matmul_for_upper_triangular_matrices',
    'N': 4096,
}


def create_inputs(dtype=jnp.float32):
    n = CONFIG['N']
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    A = jnp.triu(jax.random.uniform(k1, (n, n), dtype=dtype))
    B = jnp.triu(jax.random.uniform(k2, (n, n), dtype=dtype))
    return (A, B)


def workload(A, B):
    if A.shape[0] != A.shape[1]:
        n = min(A.shape[0], A.shape[1])
        A = A[:n, :n]
        B = B[:n, :n]
    return jnp.triu(jnp.matmul(A, B))
