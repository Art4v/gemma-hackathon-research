"""EqualConv2dSame: replicate pad -> conv2d with equalized-lr weight scaling (w / sqrt(fan_in)) + bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'EqualConv2dSame_8020',
    'batch': 16,
    'in_channel': 128,
    'out_channel': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 1,
    'padding': 0,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(kx, (c['batch'], c['in_channel'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['out_channel'], c['in_channel'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channel'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    pad = CONFIG['padding']
    stride = CONFIG['stride']
    _, cin, kh, kw = w.shape
    scale = 1.0 / jnp.sqrt(jnp.asarray(cin * kh * kw, dtype=jnp.float32))
    x = jnp.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='edge')
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(w * scale.astype(w.dtype), (2, 3, 1, 0))
    out = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(stride, stride),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    out = out + b
    return jnp.transpose(out, (0, 3, 1, 2))
