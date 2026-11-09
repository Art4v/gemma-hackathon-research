"""[m1 M2_second_output variant of kb/l1/22_Tanh] Elementwise y = tanh(x), also returning the per-row absmax of y (a quantization scale)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '22_Tanh_m1_aux',
    'batch_size': 4096,
    'dim': 16384,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch_size'], CONFIG['dim']), dtype=dtype)
    return (x,)


def workload(x):
    y = jnp.tanh(x)
    row_absmax = jnp.max(jnp.abs(y), axis=-1)
    return y, row_absmax
