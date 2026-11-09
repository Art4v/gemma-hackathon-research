"""FPNOutput: Conv2d 3x3 (bias, pad 1) -> Conv2d 1x1 (no bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'FPNOutput_808',
    'batch': 8,
    'in_chan': 128,
    'mid_chan': 256,
    'n_classes': 128,
    'height': 128,
    'width': 128,
    'ks': 3,
    'padding': 1,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    kx, k1, kb, k2 = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(kx, (c['batch'], c['in_chan'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(k1, (c['mid_chan'], c['in_chan'], c['ks'], c['ks']), dtype) * 0.02
    b1 = jax.random.normal(kb, (c['mid_chan'],), dtype) * 0.02
    w2 = jax.random.normal(k2, (c['n_classes'], c['mid_chan'], 1, 1), dtype) * 0.02
    return x, w1, b1, w2


def _conv_nhwc(x, w_oihw, pad):
    w = jnp.transpose(w_oihw, (2, 3, 1, 0))  # OIHW -> HWIO
    return lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))


def workload(x, w1, b1, w2):
    h = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
    h = _conv_nhwc(h, w1, CONFIG['padding']) + b1
    h = _conv_nhwc(h, w2, 0)
    return jnp.transpose(h, (0, 3, 1, 2))  # NHWC -> NCHW
