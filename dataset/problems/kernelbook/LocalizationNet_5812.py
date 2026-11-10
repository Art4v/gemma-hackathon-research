"""LocalizationNet: maxpool -> conv3x3 -> relu -> maxpool -> flatten -> linear -> relu."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'LocalizationNet_5812',
    'batch': 8,
    'inplanes': 4,
    'height': 512,
    'width': 512,
    'conv_channels': 64,
    'kernel_size': 3,
    'conv_stride': 1,
    'conv_padding': 1,
    'pool_window': 2,
    'pool_stride': 2,
    'factor': 4,
    'out_features': 30,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_cw, k_cb, k_lw, k_lb = jax.random.split(key, 5)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['inplanes'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    scale = CONFIG['weight_scale']
    conv_w = jax.random.normal(
        k_cw,
        (CONFIG['kernel_size'], CONFIG['kernel_size'], CONFIG['inplanes'], CONFIG['conv_channels']),
        dtype,
    ) * scale
    conv_b = jax.random.normal(k_cb, (CONFIG['conv_channels'],), dtype) * scale
    linear_in = CONFIG['conv_channels'] * (CONFIG['height'] // CONFIG['factor']) * (CONFIG['width'] // CONFIG['factor'])
    linear_w = jax.random.normal(k_lw, (linear_in, CONFIG['out_features']), dtype) * scale
    linear_b = jax.random.normal(k_lb, (CONFIG['out_features'],), dtype) * scale
    return x, conv_w, conv_b, linear_w, linear_b


def _max_pool_nchw(x):
    window = CONFIG['pool_window']
    stride = CONFIG['pool_stride']
    return lax.reduce_window(
        x,
        -jnp.inf,
        lax.max,
        (1, 1, window, window),
        (1, 1, stride, stride),
        'VALID',
    )


def workload(x, conv_w, conv_b, linear_w, linear_b):
    y = _max_pool_nchw(x)
    y = jnp.transpose(y, (0, 2, 3, 1))
    stride = CONFIG['conv_stride']
    padding = CONFIG['conv_padding']
    y = lax.conv_general_dilated(
        y,
        conv_w,
        window_strides=(stride, stride),
        padding=[(padding, padding), (padding, padding)],
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + conv_b
    y = jnp.maximum(y, 0.0)
    y = jnp.transpose(y, (0, 3, 1, 2))
    y = _max_pool_nchw(y)
    y = jnp.reshape(y, (-1, linear_w.shape[0]))
    y = y @ linear_w + linear_b
    return jnp.maximum(y, 0.0)
