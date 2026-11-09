"""[bf16 variant] CompositeActivation: atan -> concat([x / 0.67, x * x / 0.6], axis=1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CompositeActivation_1340',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'concat_axis': 1,
    'lin_div': 0.67,
    'sq_div': 0.6,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    a = jnp.arctan(x)
    return jnp.concatenate(
        [a / CONFIG['lin_div'], a * a / CONFIG['sq_div']],
        axis=CONFIG['concat_axis'],
    )
