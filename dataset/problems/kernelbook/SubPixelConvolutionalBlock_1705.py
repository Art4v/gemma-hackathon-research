"""Conv2d -> PixelShuffle -> PReLU (SubPixelConvolutionalBlock)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SubPixelConvolutionalBlock_1705',
    'batch': 8,
    'n_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'padding': 1,
    'scaling_factor': 3,
    'weight_scale': 0.02,
    'prelu_init': 0.25,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    n = CONFIG['batch']
    c = CONFIG['n_channels']
    h = CONFIG['height']
    w = CONFIG['width']
    ks = CONFIG['kernel_size']
    out_c = c * CONFIG['scaling_factor'] ** 2
    x = jax.random.uniform(k_x, (n, c, h, w), dtype)
    weight = jax.random.normal(k_w, (out_c, c, ks, ks), dtype) * CONFIG['weight_scale']
    bias = jax.random.normal(k_b, (out_c,), dtype) * CONFIG['weight_scale']
    alpha = jnp.full((1,), CONFIG['prelu_init'], dtype)
    return x, weight, bias, alpha


def workload(x, weight, bias, alpha):
    p = CONFIG['padding']
    r = CONFIG['scaling_factor']
    y = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        y,
        weight,
        window_strides=(1, 1),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    y = y + bias
    y = jnp.transpose(y, (0, 3, 1, 2))
    n, c, h, w = y.shape
    c_out = c // (r * r)
    y = jnp.reshape(y, (n, c_out, r, r, h, w))
    y = jnp.transpose(y, (0, 1, 4, 2, 5, 3))
    y = jnp.reshape(y, (n, c_out, h * r, w * r))
    return jnp.where(y >= 0, y, alpha * y)
