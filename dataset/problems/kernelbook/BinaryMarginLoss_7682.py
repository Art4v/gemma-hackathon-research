"""[bf16 variant] BinaryMarginLoss: elementwise logaddexp(1.0, margin - x)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'BinaryMarginLoss_7682',
    'shape': (32, 128, 128, 128),
    'margin': 0.5,
    'const': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    x = jax.random.uniform(jax.random.key(0), CONFIG['shape'], dtype)
    return (x,)


def workload(x):
    c = jnp.asarray(CONFIG['const'], x.dtype)
    return jnp.logaddexp(c, CONFIG['margin'] - x)
