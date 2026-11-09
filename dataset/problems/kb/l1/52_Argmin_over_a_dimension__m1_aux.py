"""[m1 M2_second_output variant of kb/l1/52_Argmin_over_a_dimension] Argmin over dim 1, also returning the min values at those indices (as torch.min(x, dim) does)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '52_Argmin_over_a_dimension_m1_aux',
    'batch_size': 16,
    'dim1': 4096,
    'dim2': 2048,
    'dim': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch_size'], CONFIG['dim1'], CONFIG['dim2']), dtype=dtype
    )
    return (x,)


def workload(x):
    dim = CONFIG['dim']
    return jnp.argmin(x, axis=dim), jnp.min(x, axis=dim)
