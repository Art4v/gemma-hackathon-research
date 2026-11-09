"""[bf16 variant] AdaIN: two Linear affines on style w, per-channel spatial centering minus rsqrt(var) (faithful to source), then scale and shift."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'AdaIN_7467',
    'batch': 16,
    'channels_in': 512,
    'channels_out': 512,
    'height': 64,
    'width': 64,
    'norm': True,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    kx, kw, k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 6)
    x = jax.random.uniform(kx, (c['batch'], c['channels_out'], c['height'], c['width']), dtype)
    w = jax.random.uniform(kw, (c['batch'], c['channels_in']), dtype)
    w_scale = jax.random.normal(k1, (c['channels_in'], c['channels_out']), dtype) * 0.02
    b_scale = jax.random.normal(k2, (c['channels_out'],), dtype) * 0.02
    w_bias = jax.random.normal(k3, (c['channels_in'], c['channels_out']), dtype) * 0.02
    b_bias = jax.random.normal(k4, (c['channels_out'],), dtype) * 0.02
    return x, w, w_scale, b_scale, w_bias, b_bias


def workload(x, w, w_scale, b_scale, w_bias, b_bias):
    ys = (w @ w_scale + b_scale)[..., None]
    yb = (w @ w_bias + b_bias)[..., None]
    xm = jnp.reshape(x, (x.shape[0], x.shape[1], -1))
    if CONFIG['norm']:
        xm_centered = xm - jnp.mean(xm, axis=2, keepdims=True)
        xm_std_rev = lax.rsqrt(jnp.mean(xm_centered * xm_centered, axis=2, keepdims=True))
        # Subtraction (not multiplication) is what the PyTorch source does; kept for parity.
        xm_norm = xm_centered - xm_std_rev
    else:
        xm_norm = xm
    return jnp.reshape(xm_norm * ys + yb, x.shape)
