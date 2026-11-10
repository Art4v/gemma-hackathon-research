"""[bf16 variant] Cast to float32 -> sqrt -> divide by constant -> floor."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'TracedModule_2755',
    'batch': 16,
    'dim_1': 128,
    'dim_2': 128,
    'dim_3': 256,
    'divisor': 5.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['dim_1'], CONFIG['dim_2'], CONFIG['dim_3'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    x = x.astype(jnp.float32)
    return jnp.floor(jnp.sqrt(x) / CONFIG['divisor'])
