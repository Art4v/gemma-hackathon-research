"""Merge: 1x1 Conv2d with bias (no ReLU, no BatchNorm)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Merge_13453',
    'batch': 16,
    'x_dim': 128,
    'y_dim': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 1,
    'stride': 1,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    c = CONFIG
    x = jax.random.uniform(k_x, (c['batch'], c['x_dim'], c['height'], c['width']), dtype)
    w = jax.random.normal(k_w, (c['kernel_size'], c['kernel_size'], c['x_dim'], c['y_dim']), dtype) * 0.02
    b = jax.random.normal(k_b, (c['y_dim'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    pad = (w.shape[0] - 1) // 2
    stride = CONFIG['stride']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x_nhwc, w,
        window_strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + b
    return jnp.transpose(y, (0, 3, 1, 2))
