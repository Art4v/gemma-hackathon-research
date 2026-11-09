"""[bf16 variant] RMSPool: x**2 -> avg_pool2d(kernel_size, stride) -> sqrt."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'RMSPool_1724',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 4,
    'stride': 1,
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
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    summed = lax.reduce_window(
        x * x,
        jnp.array(0, x.dtype),
        lax.add,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding='VALID',
    )
    return jnp.sqrt(summed / (k * k))
