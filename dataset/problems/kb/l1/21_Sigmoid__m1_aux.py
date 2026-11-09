"""[m1 M2_second_output variant of kb/l1/21_Sigmoid] Sigmoid y = sigmoid(x), also returning the per-row mean gate activation of y."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '21_Sigmoid_m1_aux',
    'batch': 4096,
    'dim': 16384,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch'], CONFIG['dim']), dtype=dtype)
    return (x,)


def workload(x):
    y = jax.nn.sigmoid(x)
    row_mean = jnp.mean(y, axis=-1)
    return y, row_mean
