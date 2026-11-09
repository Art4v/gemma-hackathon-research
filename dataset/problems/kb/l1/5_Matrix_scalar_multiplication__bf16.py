"""[bf16 variant] Matrix-scalar multiplication: C = A * s."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '5_Matrix_scalar_multiplication',
    'M': 8192,
    'N': 8192,
    's': 3.14,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    A = jax.random.uniform(key, (CONFIG['M'], CONFIG['N']), dtype=dtype)
    s = jnp.asarray(CONFIG['s'], dtype=dtype)
    return (A, s)


def workload(A, s):
    return A * s.astype(A.dtype)
