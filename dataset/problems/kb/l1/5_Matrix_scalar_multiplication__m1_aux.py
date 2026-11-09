"""[m1 M2_second_output variant of kb/l1/5_Matrix_scalar_multiplication] C = A * s, also returning the per-row absmax of C (a per-row quantization scale)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '5_Matrix_scalar_multiplication_m1_aux',
    'M': 8192,
    'N': 8192,
    's': 3.14,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    A = jax.random.uniform(key, (CONFIG['M'], CONFIG['N']), dtype=dtype)
    s = jnp.asarray(CONFIG['s'], dtype=dtype)
    return (A, s)


def workload(A, s):
    C = A * s.astype(A.dtype)
    return C, jnp.max(jnp.abs(C), axis=1, keepdims=True)
