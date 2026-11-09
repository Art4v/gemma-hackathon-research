"""[bf16 variant] Zero pad (1 px) -> MaxPool2d(3, stride=2)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MaxPool_3293',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'pad': 1,
    'kernel_size': 3,
    'stride': 2,
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
    p = CONFIG['pad']
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    # F.pad pads with constant 0, not -inf, so pad values can win the max.
    xp = jnp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    return lax.reduce_window(
        xp,
        jnp.array(-jnp.inf, dtype=x.dtype),
        lax.max,
        (1, 1, k, k),
        (1, 1, s, s),
        'VALID',
    )
