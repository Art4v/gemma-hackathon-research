"""[m1 M2_second_output variant of kb/l1/32_HardTanh] HardTanh clip(x, -1, 1), also returning the per-channel sum of the clipped output over the batch rows."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '32_HardTanh_m1_aux',
    'batch_size': 512,
    'dim': 131072,
    'min_val': -1.0,
    'max_val': 1.0,
    'input_low': -2.0,
    'input_high': 2.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch_size'], CONFIG['dim']),
        dtype=dtype,
        minval=CONFIG['input_low'],
        maxval=CONFIG['input_high'],
    )
    return (x,)


def workload(x):
    y = jnp.clip(x, CONFIG['min_val'], CONFIG['max_val'])
    channel_sum = jnp.sum(y, axis=0)
    return y, channel_sum
