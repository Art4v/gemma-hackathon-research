"""[bf16 variant] MaxPool1d (kernel 8, stride 1, padding 4, dilation 3) over (N, C, L)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '41_Max_Pooling_1D',
    'batch_size': 16,
    'features': 128,
    'sequence_length': 65536,
    'kernel_size': 8,
    'stride': 1,
    'padding': 4,
    'dilation': 3,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch_size'], CONFIG['features'], CONFIG['sequence_length']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride'] if CONFIG['stride'] is not None else k
    p = CONFIG['padding']
    d = CONFIG['dilation']
    init = jnp.array(-jnp.inf, dtype=x.dtype)
    return lax.reduce_window(
        x,
        init,
        lax.max,
        window_dimensions=(1, 1, k),
        window_strides=(1, 1, s),
        padding=((0, 0), (0, 0), (p, p)),
        window_dilation=(1, 1, d),
    )
