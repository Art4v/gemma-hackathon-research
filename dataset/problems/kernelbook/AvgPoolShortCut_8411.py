"""AvgPoolShortCut: parity-selected avg_pool2d followed by zero-channel padding concatenated on the channel axis."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'AvgPoolShortCut_8411',
    'batch': 8,
    'in_c': 128,
    'out_c': 128,
    'height': 256,
    'width': 256,
    'stride': 1,
    'odd_kernel': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['in_c'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    return (x,)


def workload(x):
    stride = CONFIG['stride']
    kernel = CONFIG['odd_kernel'] if x.shape[2] % 2 != 0 else stride
    summed = lax.reduce_window(
        x,
        0.0,
        lax.add,
        (1, 1, kernel, kernel),
        (1, 1, stride, stride),
        'VALID',
    )
    pooled = summed / (kernel * kernel)
    pad = jnp.zeros(
        (pooled.shape[0], CONFIG['out_c'] - CONFIG['in_c'], pooled.shape[2], pooled.shape[3]),
        pooled.dtype,
    )
    return jnp.concatenate((pooled, pad), axis=1)
