"""Iter_Downsample: two 2x2 maxpools, then four chained 2x2 maxpools (pad 1,0,1,0) returning each stage."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Iter_Downsample_16659',
    'batch': 16,
    'channels': 64,
    'height': 512,
    'width': 512,
    'kernel': 2,
    'stride': 2,
    'init_paddings': (0, 0),
    'stage_paddings': (1, 0, 1, 0),
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype=dtype,
    )
    return (x,)


def _maxpool(x, pad):
    k, s = CONFIG['kernel'], CONFIG['stride']
    return lax.reduce_window(
        x,
        jnp.array(-jnp.inf, dtype=x.dtype),
        lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding=((0, 0), (0, 0), (pad, pad), (pad, pad)),
    )


def workload(x):
    for pad in CONFIG['init_paddings']:
        x = _maxpool(x, pad)
    outs = []
    for pad in CONFIG['stage_paddings']:
        x = _maxpool(x, pad)
        outs.append(x)
    return tuple(outs)
