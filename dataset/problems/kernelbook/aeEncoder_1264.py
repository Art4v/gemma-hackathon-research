"""aeEncoder: 5x (Conv2d k4 s2 p1 -> ReLU)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'aeEncoder_1264',
    'batch': 32,
    'channel': 4,
    'capacity': 32,
    'height': 512,
    'width': 512,
    'kernel_size': 4,
    'stride': 2,
    'padding': 1,
    'num_layers': 5,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 1 + 2 * c['num_layers'])
    x = jax.random.uniform(keys[0], (c['batch'], c['channel'], c['height'], c['width']), dtype)
    k = c['kernel_size']
    chans = [c['channel']] + [c['capacity'] * (2 ** i) for i in range(c['num_layers'])]
    params = []
    for i in range(c['num_layers']):
        w = jax.random.normal(keys[1 + 2 * i], (k, k, chans[i], chans[i + 1]), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (chans[i + 1],), dtype) * 0.02
        params += [w, b]
    return (x, *params)


def workload(x, *params):
    s, p = CONFIG['stride'], CONFIG['padding']
    h = jnp.transpose(x, (0, 2, 3, 1))
    for w, b in zip(params[0::2], params[1::2]):
        h = lax.conv_general_dilated(
            h, w, window_strides=(s, s), padding=((p, p), (p, p)),
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
        h = jax.nn.relu(h + b)
    return jnp.transpose(h, (0, 3, 1, 2))
