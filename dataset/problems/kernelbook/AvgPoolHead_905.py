"""AvgPool2d(fea_map_size, stride=1) -> flatten -> Linear (AvgPoolHead)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'AvgPoolHead_905',
    'batch': 256,
    'in_channels': 2048,
    'out_channels': 1024,
    'fea_map_size': 7,
    'height': 7,
    'width': 7,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    c = CONFIG
    x = jax.random.uniform(kx, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['in_channels'], c['out_channels']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    n, ch, h, wd = x.shape
    # Get the pool window from the shapes: fc in_features = C * oh * ow, and the
    # square output side is oh = H - k + 1.
    pos = w.shape[0] // ch
    side = int(round(pos ** 0.5))
    kh = h - side + 1
    kw = wd - side + 1
    s = lax.reduce_window(x, jnp.array(0, x.dtype), lax.add,
                          (1, 1, kh, kw), (1, 1, 1, 1), 'VALID')
    pooled = s / (kh * kw)
    flat = pooled.reshape(n, -1)
    return flat @ w + b
