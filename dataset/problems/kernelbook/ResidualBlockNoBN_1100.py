"""ResidualBlockNoBN: conv3x3+bias -> ReLU -> conv3x3+bias -> add identity shortcut -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ResidualBlockNoBN_1100',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w1, k_b1, k_w2, k_b2 = jax.random.split(jax.random.key(0), 5)
    c_in = CONFIG['in_channels']
    c_out = CONFIG['out_channels']
    k = CONFIG['kernel_size']
    x = jax.random.uniform(k_x, (CONFIG['batch'], c_in, CONFIG['height'], CONFIG['width']), dtype)
    w1 = jax.random.normal(k_w1, (c_out, c_in, k, k), dtype) * 0.02
    b1 = jax.random.normal(k_b1, (c_out,), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (c_out, c_out, k, k), dtype) * 0.02
    b2 = jax.random.normal(k_b2, (c_out,), dtype) * 0.02
    return x, w1, b1, w2, b2


def _conv(x, w, b):
    s = CONFIG['stride']
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x,
        w,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    return y + b


def workload(x, w1, b1, w2, b2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    out = jax.nn.relu(_conv(h, w1, b1))
    out = _conv(out, w2, b2)
    out = jax.nn.relu(out + h)
    return jnp.transpose(out, (0, 3, 1, 2))
