"""MaxPool2d (kernel 4, stride 4) over H, W of a channels-last (*batch, H, W, C) tensor."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MaxPool2d_6395',
    'batch': 8,
    'height': 256,
    'width': 256,
    'channels': 128,
    'kernel_size': 4,
    'stride': 4,
    'padding': 0,
    'dilation': 1,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    x = jax.random.uniform(
        key,
        (CONFIG['batch'], CONFIG['height'], CONFIG['width'], CONFIG['channels']),
        dtype=dtype,
    )
    return (x,)


def workload(x):
    *batch, h, w, c = x.shape
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    p = CONFIG['padding']
    d = CONFIG['dilation']
    xn = x.reshape(-1, h, w, c)
    # Input is already NHWC, so torch's permute to NCHW and back is skipped.
    y = lax.reduce_window(
        xn,
        jnp.array(-jnp.inf, dtype=x.dtype),
        lax.max,
        window_dimensions=(1, k, k, 1),
        window_strides=(1, s, s, 1),
        padding=((0, 0), (p, p), (p, p), (0, 0)),
        window_dilation=(1, d, d, 1),
    )
    _, nh, nw, _ = y.shape
    return y.reshape(*batch, nh, nw, c)
