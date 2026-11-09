"""ConvSqu: Conv2d (no bias, autopad) -> Mish (x * tanh(softplus(x)))."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConvSqu_2425',
    'batch': 16,
    'c1': 128,
    'c2': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 1,
    'stride': 1,
    'groups': 1,
}


def create_inputs(dtype=jnp.float32):
    kx, kw = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        kx, (CONFIG['batch'], CONFIG['c1'], CONFIG['height'], CONFIG['width']), dtype)
    k = CONFIG['kernel_size']
    w = jax.random.normal(
        kw, (k, k, CONFIG['c1'] // CONFIG['groups'], CONFIG['c2']), dtype) * 0.02
    return x, w


def workload(x, w):
    k = w.shape[0]
    p = k // 2  # autopad
    s = CONFIG['stride']
    y = lax.conv_general_dilated(
        jnp.transpose(x, (0, 2, 3, 1)), w,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=CONFIG['groups'],
    )
    y = y * jnp.tanh(jax.nn.softplus(y))
    return jnp.transpose(y, (0, 3, 1, 2))
