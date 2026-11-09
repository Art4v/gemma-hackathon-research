"""[bf16 variant] RollLayer: roll x[:, :-1] along the last axis by `shift`, then concat the untouched last column."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'RollLayer_5620',
    'batch': 8192,
    'features': 8192,
    'shift': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(key, (CONFIG['batch'], CONFIG['features']), dtype)
    return (x,)


def workload(x):
    rolled = jnp.roll(x[:, :-1], CONFIG['shift'], axis=-1)
    return jnp.concatenate((rolled, x[:, -1:]), axis=-1)
