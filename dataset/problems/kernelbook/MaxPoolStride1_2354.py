"""[bf16 variant] Zero pad (k-1)//2 on H and W, then MaxPool2d(kernel_size=k, stride=1)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MaxPoolStride1_2354',
    'batch': 16,
    'channels': 64,
    'height': 256,
    'width': 256,
    'kernel_size': 4,
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
    # Matches int(pad / 2) in the torch module: floor((k - 1) / 2).
    p = (k - 1) // 2
    padded = jnp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant', constant_values=0)
    return lax.reduce_window(
        padded,
        jnp.array(-jnp.inf, dtype=x.dtype),
        lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, 1, 1),
        padding='VALID',
    )
