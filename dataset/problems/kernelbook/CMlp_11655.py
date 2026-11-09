"""[bf16 variant] CMlp: Conv3d 1x1x1 -> GELU (exact) -> Conv3d 1x1x1 (dropout is identity)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'CMlp_11655',
    'batch': 8,
    'in_features': 128,
    'hidden_features': 128,
    'out_features': 128,
    'depth': 16,
    'height': 64,
    'width': 64,
}

_DN = ('NDHWC', 'DHWIO', 'NDHWC')


def create_inputs(dtype=jnp.bfloat16):
    k = jax.random.split(jax.random.key(0), 5)
    c = CONFIG
    x = jax.random.uniform(
        k[0], (c['batch'], c['in_features'], c['depth'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(k[1], (1, 1, 1, c['in_features'], c['hidden_features']), dtype) * 0.02
    b1 = jax.random.normal(k[2], (c['hidden_features'],), dtype) * 0.02
    w2 = jax.random.normal(k[3], (1, 1, 1, c['hidden_features'], c['out_features']), dtype) * 0.02
    b2 = jax.random.normal(k[4], (c['out_features'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def _conv1x1x1(h, w, b):
    return lax.conv_general_dilated(h, w, (1, 1, 1), 'VALID', dimension_numbers=_DN) + b


def workload(x, w1, b1, w2, b2):
    # torch Conv3d accepts unbatched (C, D, H, W) input; mirror that.
    unbatched = x.ndim == 4
    if unbatched:
        x = x[None]
    h = jnp.transpose(x, (0, 2, 3, 4, 1))
    h = _conv1x1x1(h, w1, b1)
    h = jax.nn.gelu(h, approximate=False)
    h = _conv1x1x1(h, w2, b2)
    out = jnp.transpose(h, (0, 4, 1, 2, 3))
    return out[0] if unbatched else out
