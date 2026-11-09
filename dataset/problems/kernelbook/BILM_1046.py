"""[bf16 variant] BILM: sigmoid -> maxpool3x3(sig) + maxpool3x3(-sig) -> feat * sum."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'BILM_1046',
    'batch': 16,
    'channels': 64,
    'height': 512,
    'width': 512,
    'pool_kernel': 3,
    'pool_stride': 1,
    'pool_padding': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    shape = (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width'])
    feat = jax.random.uniform(key, shape, dtype)
    return (feat,)


def _maxpool_nchw(x):
    k = CONFIG['pool_kernel']
    s = CONFIG['pool_stride']
    p = CONFIG['pool_padding']
    # PyTorch MaxPool2d pads with -inf, so padded cells never win the max.
    return lax.reduce_window(
        x,
        jnp.array(-jnp.inf, x.dtype),
        lax.max,
        window_dimensions=(1, 1, k, k),
        window_strides=(1, 1, s, s),
        padding=((0, 0), (0, 0), (p, p), (p, p)),
    )


def workload(feat):
    pos_sig = jax.nn.sigmoid(feat)
    neg_sig = -1 * pos_sig
    sum_sig = _maxpool_nchw(pos_sig) + _maxpool_nchw(neg_sig)
    return feat * sum_sig
