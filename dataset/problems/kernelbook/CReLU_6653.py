"""[bf16 variant] CReLU: concat(relu(x), relu(-x)) along the channel dim."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CReLU_6653',
    'batch': 16,
    'channels': 128,
    'height': 128,
    'width': 128,
    'concat_axis': 1,
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
    return jnp.concatenate(
        (jax.nn.relu(x), jax.nn.relu(-x)), axis=CONFIG['concat_axis']
    )
