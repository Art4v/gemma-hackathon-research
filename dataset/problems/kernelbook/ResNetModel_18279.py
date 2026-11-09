"""[bf16 variant] ResNetModel_18279: conv/relu/maxpool + avgpool-identity concat x4, conv5/relu, global avgpool, fc/relu, fc."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ResNetModel_18279',
    'batch': 64,
    'in_channels': 1,
    'height': 512,
    'width': 512,
    'initial_filters': 16,
    'num_outputs': 2,
    'kernel_size': 3,
    'conv1_stride': 2,
    'conv_padding': 1,
    'pool': 2,
    'identity_pool': 4,
    'fc_hidden_mult': 64,
}


def _conv_relu(x, w, b, stride):
    p = CONFIG['conv_padding']
    y = lax.conv_general_dilated(
        x, w, window_strides=(stride, stride), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
    return jax.nn.relu(y + b)


def _avg_pool(x, k):
    s = lax.reduce_window(x, 0.0, lax.add, (1, k, k, 1), (1, k, k, 1), 'VALID')
    return s / (k * k)


def _max_pool(x, k):
    return lax.reduce_window(x, -jnp.inf, lax.max, (1, k, k, 1), (1, k, k, 1), 'VALID')


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    f = c['initial_filters']
    cin = c['in_channels']
    k = c['kernel_size']
    conv_io = [
        (cin, f),
        (f + cin, 2 * f),
        (3 * f + cin, 4 * f),
        (7 * f + cin, 8 * f),
        (15 * f + cin, 16 * f),
    ]
    keys = jax.random.split(jax.random.key(0), 1 + 2 * len(conv_io) + 4)
    x = jax.random.uniform(keys[0], (c['batch'], cin, c['height'], c['width']), dtype)
    params = []
    for i, (ci, co) in enumerate(conv_io):
        params.append(jax.random.normal(keys[1 + 2 * i], (co, ci, k, k), dtype) * 0.02)
        params.append(jax.random.normal(keys[2 + 2 * i], (co,), dtype) * 0.02)
    hidden = c['fc_hidden_mult'] * f
    kf = keys[1 + 2 * len(conv_io):]
    fc1_w = jax.random.normal(kf[0], (16 * f, hidden), dtype) * 0.02
    fc1_b = jax.random.normal(kf[1], (hidden,), dtype) * 0.02
    fco_w = jax.random.normal(kf[2], (hidden, c['num_outputs']), dtype) * 0.02
    fco_b = jax.random.normal(kf[3], (c['num_outputs'],), dtype) * 0.02
    return (x, *params, fc1_w, fc1_b, fco_w, fco_b)


def workload(x, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, fc1_w, fc1_b, fco_w, fco_b):
    pool = CONFIG['pool']
    x = jnp.transpose(x, (0, 2, 3, 1))
    identity = _avg_pool(x, CONFIG['identity_pool'])
    x = _max_pool(_conv_relu(x, w1, b1, CONFIG['conv1_stride']), pool)
    x = jnp.concatenate((x, identity), axis=-1)
    for w, b in ((w2, b2), (w3, b3), (w4, b4)):
        identity = _avg_pool(x, pool)
        x = _max_pool(_conv_relu(x, w, b, 1), pool)
        x = jnp.concatenate((x, identity), axis=-1)
    x = _conv_relu(x, w5, b5, 1)
    x = jnp.mean(x, axis=(1, 2))
    x = jax.nn.relu(x @ fc1_w + fc1_b)
    return x @ fco_w + fco_b
