"""[bf16 variant] GroupNorm -> ReLU -> ConcatConv2d(t) -> GroupNorm -> ReLU -> ConcatConv2d(t) -> GroupNorm (ODE func)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ODEfunc_1191',
    'batch': 8,
    'dim': 128,
    'height': 128,
    'width': 128,
    'ksize': 3,
    'padding': 1,
    'max_groups': 32,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    b, c = CONFIG['batch'], CONFIG['dim']
    h, w, k = CONFIG['height'], CONFIG['width'], CONFIG['ksize']
    keys = jax.random.split(jax.random.key(0), 6)
    t = jax.random.uniform(keys[0], (b, 1, h, w), dtype)
    x = jax.random.uniform(keys[1], (b, c, h, w), dtype)
    w1 = jax.random.normal(keys[2], (c, c + 1, k, k), dtype) * 0.02
    cb1 = jax.random.normal(keys[3], (c,), dtype) * 0.02
    w2 = jax.random.normal(keys[4], (c, c + 1, k, k), dtype) * 0.02
    cb2 = jax.random.normal(keys[5], (c,), dtype) * 0.02
    ones = jnp.ones((c,), dtype)
    zeros = jnp.zeros((c,), dtype)
    return (t, x, ones, zeros, w1, cb1, ones, zeros, w2, cb2, ones, zeros)


def _group_norm(x, gamma, beta):
    b, c, h, w = x.shape
    g = min(CONFIG['max_groups'], c)
    xr = x.reshape(b, g, c // g, h, w)
    mean = jnp.mean(xr, axis=(2, 3, 4), keepdims=True)
    var = jnp.var(xr, axis=(2, 3, 4), keepdims=True)
    xr = (xr - mean) * lax.rsqrt(var + CONFIG['eps'])
    return xr.reshape(b, c, h, w) * gamma[None, :, None, None] + beta[None, :, None, None]


def _concat_conv(t, x, w, bias):
    tt = jnp.ones_like(x[:, :1, :, :]) * t
    ttx = jnp.concatenate([tt, x], axis=1)
    ttx_nhwc = jnp.transpose(ttx, (0, 2, 3, 1))
    w_hwio = jnp.transpose(w, (2, 3, 1, 0))
    p = CONFIG['padding']
    out = lax.conv_general_dilated(
        ttx_nhwc, w_hwio, window_strides=(1, 1), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    out = out + bias[None, None, None, :]
    return jnp.transpose(out, (0, 3, 1, 2))


def workload(t, x, g1, b1, w1, cb1, g2, b2, w2, cb2, g3, b3):
    out = jax.nn.relu(_group_norm(x, g1, b1))
    out = _concat_conv(t, out, w1, cb1)
    out = jax.nn.relu(_group_norm(out, g2, b2))
    out = _concat_conv(t, out, w2, cb2)
    return _group_norm(out, g3, b3)
