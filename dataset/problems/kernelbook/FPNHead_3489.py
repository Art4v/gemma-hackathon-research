"""FPNHead: Conv2d 3x3 -> ReLU -> Conv2d 3x3 -> ReLU (NCHW interface, no bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'FPNHead_3489',
    'batch': 8,
    'num_in': 128,
    'num_mid': 128,
    'num_out': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'padding': 1,
    'stride': 1,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w0, k_w1 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['num_in'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    ks = CONFIG['kernel_size']
    w0 = jax.random.normal(k_w0, (CONFIG['num_mid'], CONFIG['num_in'], ks, ks), dtype) * 0.02
    w1 = jax.random.normal(k_w1, (CONFIG['num_out'], CONFIG['num_mid'], ks, ks), dtype) * 0.02
    return x, w0, w1


def _conv_relu(x_nhwc, w_oihw):
    kernel = jnp.transpose(w_oihw, (2, 3, 1, 0))
    p = CONFIG['padding']
    s = CONFIG['stride']
    y = lax.conv_general_dilated(
        x_nhwc,
        kernel,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return jax.nn.relu(y)


def workload(x, w0, w1):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _conv_relu(h, w0)
    h = _conv_relu(h, w1)
    return jnp.transpose(h, (0, 3, 1, 2))
