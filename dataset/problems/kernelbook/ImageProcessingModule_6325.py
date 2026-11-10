"""[bf16 variant] Conv2d(k=7, stride=7, bias) -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ImageProcessingModule_6325',
    'batch': 16,
    'in_channels': 3,
    'height': 896,
    'width': 896,
    'n_filters': 128,
    'kernel_size': 7,
    'stride': 7,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']), dtype)
    k = CONFIG['kernel_size']
    w = jax.random.normal(k_w, (k, k, CONFIG['in_channels'], CONFIG['n_filters']), dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['n_filters'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    s = CONFIG['stride']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x_nhwc, w,
        window_strides=(s, s),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = jax.nn.relu(y + b)
    return jnp.transpose(y, (0, 3, 1, 2))
