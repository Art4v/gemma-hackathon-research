"""[m1 M2_second_output variant of kb/l1/51_Argmax_over_a_dimension] Argmax over dim 1 of a 3D tensor, also returning the max values at those indices (as torch.max(x, dim) does)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '51_Argmax_over_a_dimension_m1_aux',
    'batch_size': 16,
    'dim1': 2048,
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
    return jnp.argmax(x, axis=dim), jnp.max(x, axis=dim)
