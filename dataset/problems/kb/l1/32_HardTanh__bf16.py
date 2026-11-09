"""[bf16 variant] HardTanh activation: clip(x, -1, 1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': '32_HardTanh',
    'batch_size': 512,
    'dim': 131072,
    'min_val': -1.0,
    'max_val': 1.0,
    'input_low': -2.0,
    'input_high': 2.0,
}


def create_inputs(dtype=jnp.bfloat16):
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
    return jnp.clip(x, CONFIG['min_val'], CONFIG['max_val'])
