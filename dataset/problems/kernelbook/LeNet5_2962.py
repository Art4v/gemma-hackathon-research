"""[bf16 variant] LeNet5: conv5x5 -> maxpool2 -> conv5x5 -> maxpool2 -> flatten -> linear -> linear -> linear."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'LeNet5_2962',
    'batch': 16384,
    'in_channels': 1,
    'height': 32,
    'width': 32,
    'conv1_out': 6,
    'conv2_out': 16,
    'kernel_size': 5,
    'pool_size': 2,
    'fc1_out': 120,
    'fc2_out': 84,
    'fc3_out': 10,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k = c['kernel_size']
    keys = jax.random.split(jax.random.key(0), 11)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(keys[1], (k, k, c['in_channels'], c['conv1_out']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['conv1_out'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (k, k, c['conv1_out'], c['conv2_out']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['conv2_out'],), dtype) * 0.02
    h = ((c['height'] - k + 1) // c['pool_size'] - k + 1) // c['pool_size']
    w = ((c['width'] - k + 1) // c['pool_size'] - k + 1) // c['pool_size']
    flat = c['conv2_out'] * h * w
    fw1 = jax.random.normal(keys[5], (flat, c['fc1_out']), dtype) * 0.02
    fb1 = jax.random.normal(keys[6], (c['fc1_out'],), dtype) * 0.02
    fw2 = jax.random.normal(keys[7], (c['fc1_out'], c['fc2_out']), dtype) * 0.02
    fb2 = jax.random.normal(keys[8], (c['fc2_out'],), dtype) * 0.02
    fw3 = jax.random.normal(keys[9], (c['fc2_out'], c['fc3_out']), dtype) * 0.02
    fb3 = jax.random.normal(keys[10], (c['fc3_out'],), dtype) * 0.02
    return (x, w1, b1, w2, b2, fw1, fb1, fw2, fb2, fw3, fb3)


def _conv(x, w, b):
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def _maxpool(x):
    p = CONFIG['pool_size']
    return lax.reduce_window(
        x, -jnp.inf, lax.max,
        window_dimensions=(1, p, p, 1), window_strides=(1, p, p, 1), padding='VALID')


def workload(x, w1, b1, w2, b2, fw1, fb1, fw2, fb2, fw3, fb3):
    x = jnp.transpose(x, (0, 2, 3, 1))
    x = _maxpool(_conv(x, w1, b1))
    x = _maxpool(_conv(x, w2, b2))
    # Flatten in NCHW order to match torch's view(-1, C*H*W).
    x = jnp.transpose(x, (0, 3, 1, 2)).reshape(x.shape[0], -1)
    x = x @ fw1 + fb1
    x = x @ fw2 + fb2
    return x @ fw3 + fb3
