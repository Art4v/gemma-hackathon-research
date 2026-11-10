"""ReflectionPad2d(kernel_size//2) -> Conv2d(kernel_size, stride) with bias (UpsampleConvLayer, upsample=None)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'UpsampleConvLayer_11372',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 1,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    c = CONFIG
    k = c['kernel_size']
    x = jax.random.uniform(kx, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (k, k, c['in_channels'], c['out_channels']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    p = CONFIG['kernel_size'] // 2
    s = CONFIG['stride']
    x = jnp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='reflect')
    x = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x, w, window_strides=(s, s), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    y = y + b
    return jnp.transpose(y, (0, 3, 1, 2))
