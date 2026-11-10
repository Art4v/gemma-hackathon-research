"""Conv2d (3x3, stride 1, no padding) -> ReLU -> flatten to rows of 8 -> Linear -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'BasicModel_ConvNet_One_Conv_5489',
    'batch': 8,
    'in_channels': 1,
    'conv_channels': 2,
    'height': 2048,
    'width': 2048,
    'kernel_size': 3,
    'stride': 1,
    'padding': 0,
    'fc_in': 8,
    'fc_out': 4,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_conv_w, k_conv_b, k_fc_w, k_fc_b = jax.random.split(key, 5)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    conv_w = jax.random.normal(
        k_conv_w,
        (CONFIG['conv_channels'], CONFIG['in_channels'], CONFIG['kernel_size'], CONFIG['kernel_size']),
        dtype,
    ) * CONFIG['weight_scale']
    conv_b = jax.random.normal(k_conv_b, (CONFIG['conv_channels'],), dtype) * CONFIG['weight_scale']
    fc_w = jax.random.normal(k_fc_w, (CONFIG['fc_in'], CONFIG['fc_out']), dtype) * CONFIG['weight_scale']
    fc_b = jax.random.normal(k_fc_b, (CONFIG['fc_out'],), dtype) * CONFIG['weight_scale']
    return x, conv_w, conv_b, fc_w, fc_b


def workload(x, conv_w, conv_b, fc_w, fc_b):
    pad = CONFIG['padding']
    stride = CONFIG['stride']
    y = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        y,
        conv_w,
        window_strides=(stride, stride),
        padding=[(pad, pad), (pad, pad)],
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    y = y + conv_b
    y = jnp.maximum(y, 0)
    y = jnp.transpose(y, (0, 3, 1, 2))
    y = y.reshape(-1, fc_w.shape[0])
    y = y @ fc_w + fc_b
    return jnp.maximum(y, 0)
