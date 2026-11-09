"""[bf16 variant] VGG16 trunk: 13x (conv3x3 + bias + ReLU) with 2x2 max-pools after conv 2, 4, 7, returning relu5_3."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Vgg16_13529',
    'batch': 8,
    'in_channels': 3,
    'height': 256,
    'width': 256,
    'channels': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
    'pool_after': [1, 3, 6],  # zero-based conv indices followed by max-pool
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'pool_size': 2,
    'pool_stride': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 1 + 2 * len(CONFIG['channels']))
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    k = CONFIG['kernel_size']
    params = []
    cin = CONFIG['in_channels']
    for i, cout in enumerate(CONFIG['channels']):
        w = jax.random.normal(keys[1 + 2 * i], (k, k, cin, cout), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (cout,), dtype) * 0.02
        params += [w, b]
        cin = cout
    return (x, *params)


def _conv_relu(h, w, b):
    s, p = CONFIG['stride'], CONFIG['padding']
    y = lax.conv_general_dilated(
        h, w,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return jax.nn.relu(y + b)


def _max_pool(h):
    k, s = CONFIG['pool_size'], CONFIG['pool_stride']
    return lax.reduce_window(h, -jnp.inf, lax.max, (1, k, k, 1), (1, s, s, 1), 'VALID')


def workload(x, *params):
    h = jnp.transpose(x, (0, 2, 3, 1))
    for i in range(len(params) // 2):
        h = _conv_relu(h, params[2 * i], params[2 * i + 1])
        if i in CONFIG['pool_after']:
            h = _max_pool(h)
    return jnp.transpose(h, (0, 3, 1, 2))
