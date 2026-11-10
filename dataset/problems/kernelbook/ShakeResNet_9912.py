"""Conv2d(3x3, bias) -> ReLU -> AvgPool2d(8) -> row-major NCHW reshape(-1, F) -> Linear(bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ShakeResNet_9912',
    'batch': 24,
    'in_channels': 3,
    'conv_channels': 128,
    'height': 512,
    'width': 512,
    'kernel_size': 3,
    'padding': 1,
    'stride': 1,
    'pool_size': 8,
    'num_classes': 128,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_conv_w, k_conv_b, k_fc_w = jax.random.split(key, 4)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    conv_w = jax.random.normal(
        k_conv_w,
        (CONFIG['conv_channels'], CONFIG['in_channels'], CONFIG['kernel_size'], CONFIG['kernel_size']),
        dtype,
    ) * 0.02
    conv_b = jax.random.normal(k_conv_b, (CONFIG['conv_channels'],), dtype) * 0.02
    fc_w = jax.random.normal(k_fc_w, (CONFIG['conv_channels'], CONFIG['num_classes']), dtype) * 0.02
    fc_b = jnp.zeros((CONFIG['num_classes'],), dtype)
    return x, conv_w, conv_b, fc_w, fc_b


def workload(x, conv_w, conv_b, fc_w, fc_b):
    pad = CONFIG['padding']
    stride = CONFIG['stride']
    pool = CONFIG['pool_size']

    h = jnp.transpose(x, (0, 2, 3, 1))
    kernel = jnp.transpose(conv_w, (2, 3, 1, 0))
    h = lax.conv_general_dilated(
        h,
        kernel,
        window_strides=(stride, stride),
        padding=[(pad, pad), (pad, pad)],
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    h = h + conv_b
    h = jnp.maximum(h, 0.0)
    h = lax.reduce_window(
        h,
        0.0,
        lax.add,
        (1, pool, pool, 1),
        (1, pool, pool, 1),
        'VALID',
    ) / (pool * pool)
    h = jnp.transpose(h, (0, 3, 1, 2))
    h = jnp.reshape(h, (-1, fc_w.shape[0]))
    return h @ fc_w + fc_b
