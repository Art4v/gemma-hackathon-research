"""[bf16 variant] Constant-zero TensorFlow-SAME-style dynamic pad followed by 2D max pooling (NCHW)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MaxPool2dDynamicSamePadding_11436',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': (4, 4),
    'stride': (1, 1),
    'dilation': (1, 1),
    'pad_value': 0.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    x = jax.random.uniform(key, shape, dtype)
    return (x,)


def workload(x):
    ih, iw = x.shape[-2:]
    kh, kw = CONFIG['kernel_size']
    sh, sw = CONFIG['stride']
    dh, dw = CONFIG['dilation']
    oh = -(-ih // sh)
    ow = -(-iw // sw)
    pad_h = max((oh - 1) * sh + (kh - 1) * dh + 1 - ih, 0)
    pad_w = max((ow - 1) * sw + (kw - 1) * dw + 1 - iw, 0)
    pad_width = (
        (0, 0),
        (0, 0),
        (pad_h // 2, pad_h - pad_h // 2),
        (pad_w // 2, pad_w - pad_w // 2),
    )
    padded = jnp.pad(x, pad_width, mode='constant', constant_values=CONFIG['pad_value'])
    return lax.reduce_window(
        padded,
        jnp.array(-jnp.inf, padded.dtype),
        lax.max,
        window_dimensions=(1, 1, kh, kw),
        window_strides=(1, 1, sh, sw),
        padding='VALID',
        window_dilation=(1, 1, dh, dw),
    )
