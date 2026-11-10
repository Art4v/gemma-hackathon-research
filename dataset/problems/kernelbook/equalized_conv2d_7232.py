"""Equalized-LR conv2d: input scaling -> Conv2d (no conv bias) -> per-channel bias add."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'equalized_conv2d_7232',
    'batch': 8,
    'c_in': 128,
    'c_out': 128,
    'height': 128,
    'width': 128,
    'k_size': 4,
    'stride': 1,
    'pad': 4,
    'a': 0.0,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['c_in'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w = jax.random.normal(
        k_w,
        (CONFIG['c_out'], CONFIG['c_in'], CONFIG['k_size'], CONFIG['k_size']),
        dtype,
    ) * 0.02
    b = jax.random.normal(k_b, (CONFIG['c_out'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    # w is OIHW (c_out, c_in, kH, kW), as stored by PyTorch.
    fan_in = w.shape[1] * w.shape[2] * w.shape[3]
    gain = (2.0 / (1.0 + CONFIG['a'] ** 2)) ** 0.5
    scale = gain / fan_in ** 0.5

    pad = CONFIG['pad']
    stride = CONFIG['stride']
    y = lax.conv_general_dilated(
        jnp.transpose(x * scale, (0, 2, 3, 1)),
        w,
        window_strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    y = jnp.transpose(y, (0, 3, 1, 2))
    return y + b.reshape((1, -1, 1, 1))
