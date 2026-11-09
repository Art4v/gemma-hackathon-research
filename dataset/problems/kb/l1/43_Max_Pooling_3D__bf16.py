"""[bf16 variant] MaxPool3d (kernel 3, stride 2, padding 1, dilation 3) on an NCDHW tensor."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '43_Max_Pooling_3D',
    'batch_size': 8,
    'channels': 8,
    'dim1': 128,
    'dim2': 128,
    'dim3': 128,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'dilation': 3,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch_size'], CONFIG['channels'], CONFIG['dim1'], CONFIG['dim2'], CONFIG['dim3']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    p = CONFIG['padding']
    d = CONFIG['dilation']
    init = jnp.array(-jnp.inf, dtype=x.dtype)
    return lax.reduce_window(
        x,
        init,
        lax.max,
        window_dimensions=(1, 1, k, k, k),
        window_strides=(1, 1, s, s, s),
        padding=((0, 0), (0, 0), (p, p), (p, p), (p, p)),
        window_dilation=(1, 1, d, d, d),
    )
