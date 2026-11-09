"""LeNet: conv5x5 -> maxpool2 -> conv5x5 -> maxpool2 -> flatten -> linear -> linear -> linear."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Net_36',
    'batch': 8192,
    'in_channels': 3,
    'height': 32,
    'width': 32,
    'conv1_out': 6,
    'conv2_out': 16,
    'kernel_size': 5,
    'pool_size': 2,
    'pool_stride': 2,
    'flat_features': 16 * 5 * 5,
    'fc1_out': 120,
    'fc2_out': 84,
    'fc3_out': 10,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 11)
    k = c['kernel_size']
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(keys[1], (k, k, c['in_channels'], c['conv1_out']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['conv1_out'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (k, k, c['conv1_out'], c['conv2_out']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['conv2_out'],), dtype) * 0.02
    fw1 = jax.random.normal(keys[5], (c['flat_features'], c['fc1_out']), dtype) * 0.02
    fb1 = jax.random.normal(keys[6], (c['fc1_out'],), dtype) * 0.02
    fw2 = jax.random.normal(keys[7], (c['fc1_out'], c['fc2_out']), dtype) * 0.02
    fb2 = jax.random.normal(keys[8], (c['fc2_out'],), dtype) * 0.02
    fw3 = jax.random.normal(keys[9], (c['fc2_out'], c['fc3_out']), dtype) * 0.02
    fb3 = jax.random.normal(keys[10], (c['fc3_out'],), dtype) * 0.02
    return x, w1, b1, w2, b2, fw1, fb1, fw2, fb2, fw3, fb3


def _conv(x_nhwc, w_hwio, b):
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def _maxpool(x_nhwc):
    p, s = CONFIG['pool_size'], CONFIG['pool_stride']
    return lax.reduce_window(
        x_nhwc, -jnp.inf, lax.max,
        window_dimensions=(1, p, p, 1), window_strides=(1, s, s, 1), padding='VALID')


def workload(x, w1, b1, w2, b2, fw1, fb1, fw2, fb2, fw3, fb3):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _maxpool(_conv(h, w1, b1))
    h = _maxpool(_conv(h, w2, b2))
    # Flatten in NCHW order to match torch's x.view(-1, 16*5*5).
    h = jnp.transpose(h, (0, 3, 1, 2)).reshape(-1, CONFIG['flat_features'])
    h = h @ fw1 + fb1
    h = h @ fw2 + fb2
    return h @ fw3 + fb3
