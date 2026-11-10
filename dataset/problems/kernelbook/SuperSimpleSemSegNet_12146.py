"""[bf16 variant] SuperSimpleSemSegNet: Conv2d (3x3, stride 1, padding 1) -> ReLU -> LogSoftmax over channels."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SuperSimpleSemSegNet_12146',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'softmax_axis': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w = jax.random.normal(
        k_w,
        (CONFIG['out_channels'], CONFIG['in_channels'], CONFIG['kernel_size'], CONFIG['kernel_size']),
        dtype,
    ) * CONFIG['weight_scale']
    b = jax.random.normal(k_b, (CONFIG['out_channels'],), dtype) * CONFIG['weight_scale']
    return x, w, b


def workload(x, w, b):
    # x: NCHW, w: OIHW (out, in, kh, kw), b: (out)
    stride = CONFIG['stride']
    pad = CONFIG['padding']
    y = lax.conv_general_dilated(
        jnp.transpose(x, (0, 2, 3, 1)),
        jnp.transpose(w, (2, 3, 1, 0)),
        window_strides=(stride, stride),
        padding=[(pad, pad), (pad, pad)],
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + b.reshape((1, 1, 1, -1))
    y = jnp.maximum(y, jnp.zeros((), y.dtype))
    y = jnp.transpose(y, (0, 3, 1, 2))
    return jax.nn.log_softmax(y, axis=CONFIG['softmax_axis'])
