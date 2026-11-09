"""[m1 M2_second_output variant of kb/l1/48_Mean_reduction_over_a_dimension] Mean reduction over dim 1, also returning the per-sample absmax of the reduced output (a quantization scale)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '48_Mean_reduction_over_a_dimension_m1_aux',
    'batch_size': 16,
    'dim1': 4096,
    'dim2': 1024,
    'dim': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key, (CONFIG['batch_size'], CONFIG['dim1'], CONFIG['dim2']), dtype=dtype
    )
    return (x,)


def workload(x):
    out = jnp.mean(x, axis=CONFIG['dim'])
    scale = jnp.max(jnp.abs(out), axis=-1, keepdims=True)
    return out, scale
