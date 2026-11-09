"""[m1 M1_batch variant of kb/l1/44_Average_Pooling_1D] AvgPool1d (count_include_pad=True) over NCL input, applied over a new leading axis of size 4 with the length shrunk from 65536 to 16384."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '44_Average_Pooling_1D_m1_batch',
    'outer_batch': 4,
    'batch_size': 8,
    'in_channels': 128,
    'input_length': 16384,
    'kernel_size': 8,
    'stride': 1,
    'padding': 4,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['outer_batch'], CONFIG['batch_size'], CONFIG['in_channels'], CONFIG['input_length']),
        dtype=dtype,
    )
    return (x,)


def _avg_pool(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    p = CONFIG['padding']
    summed = lax.reduce_window(
        x,
        jnp.array(0, dtype=x.dtype),
        lax.add,
        window_dimensions=(1, 1, k),
        window_strides=(1, 1, s),
        padding=((0, 0), (0, 0), (p, p)),
    )
    return summed / jnp.array(k, dtype=x.dtype)


def workload(x):
    return jax.vmap(_avg_pool)(x)
