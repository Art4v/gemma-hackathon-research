"""LunaBlock: Conv3d(3x3x3) -> ReLU -> Conv3d(3x3x3) -> ReLU -> MaxPool3d(2, 2)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'LunaBlock_688',
    'batch': 8,
    'in_channels': 128,
    'conv_channels': 128,
    'depth': 32,
    'height': 32,
    'width': 32,
    'kernel_size': 3,
    'padding': 1,
    'pool_size': 2,
    'pool_stride': 2,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = c['kernel_size']
    kx, kw1, kb1, kw2, kb2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(
        kx, (c['batch'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(kw1, (c['conv_channels'], c['in_channels'], k, k, k), dtype) * 0.02
    b1 = jax.random.normal(kb1, (c['conv_channels'],), dtype) * 0.02
    w2 = jax.random.normal(kw2, (c['conv_channels'], c['conv_channels'], k, k, k), dtype) * 0.02
    b2 = jax.random.normal(kb2, (c['conv_channels'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def _conv_relu(x, w, b):
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x, w,
        window_strides=(1, 1, 1),
        padding=((p, p), (p, p), (p, p)),
        dimension_numbers=('NDHWC', 'OIDHW', 'NDHWC'),
        precision=lax.Precision.HIGHEST,
    )
    return jax.nn.relu(y + b)


def workload(x, w1, b1, w2, b2):
    # Conv3d accepts unbatched (C, D, H, W) input; mirror that.
    unbatched = x.ndim == 4
    if unbatched:
        x = x[None]
    h = jnp.transpose(x, (0, 2, 3, 4, 1))
    h = _conv_relu(h, w1, b1)
    h = _conv_relu(h, w2, b2)
    ps, st = CONFIG['pool_size'], CONFIG['pool_stride']
    h = lax.reduce_window(
        h, -jnp.inf, lax.max,
        window_dimensions=(1, ps, ps, ps, 1),
        window_strides=(1, st, st, st, 1),
        padding='VALID',
    )
    out = jnp.transpose(h, (0, 4, 1, 2, 3))
    return out[0] if unbatched else out
