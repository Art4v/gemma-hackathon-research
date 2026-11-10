"""SqueezeInitBlock: Conv2d (stride 2, no padding, with bias) followed by ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SqueezeInitBlock_6634',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 4,
    'stride': 2,
    'padding': 0,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w = jax.random.normal(
        k_w,
        (CONFIG['out_channels'], CONFIG['in_channels'],
         CONFIG['kernel_size'], CONFIG['kernel_size']),
        dtype,
    ) * CONFIG['weight_scale']
    b = jax.random.normal(k_b, (CONFIG['out_channels'],), dtype) * CONFIG['weight_scale']
    return x, w, b


def workload(x, w, b):
    stride = CONFIG['stride']
    pad = CONFIG['padding']
    y = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        y,
        w,
        window_strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    y = y + b
    y = jnp.maximum(y, jnp.zeros((), y.dtype))
    return jnp.transpose(y, (0, 3, 1, 2))
