"""[bf16 variant] CircularPad: circular pad along H (dim -2) followed by zero pad along W (dim -1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'CircularPad_15117',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'pad': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    p = CONFIG['pad']
    x = jnp.concatenate([x[..., -p:, :], x, x[..., :p, :]], axis=-2)
    pad_width = [(0, 0)] * (x.ndim - 1) + [(p, p)]
    return jnp.pad(x, pad_width, mode='constant', constant_values=0)
