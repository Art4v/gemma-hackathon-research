"""FirstOctaveConv: (optional avgpool) -> [conv h2h(x), conv h2l(avgpool2x2(x))]."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'FirstOctaveConv_9245',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'alpha': 0.5,
    'kernel_size': 4,
    'stride': 1,
    'padding': 1,
    'dilation': 1,
    'groups': 1,
    'height': 256,
    'width': 256,
    'pool_size': 2,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k_x, k_h, k_l = jax.random.split(jax.random.key(0), 3)
    low_ch = int(c['alpha'] * c['out_channels'])
    high_ch = c['out_channels'] - low_ch
    ks = c['kernel_size']
    cin = c['in_channels'] // c['groups']
    x = jax.random.uniform(
        k_x, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w_h2h = jax.random.normal(k_h, (ks, ks, cin, high_ch), dtype) * 0.02
    w_h2l = jax.random.normal(k_l, (ks, ks, cin, low_ch), dtype) * 0.02
    return x, w_h2h, w_h2l


def _avg_pool(x):
    # NHWC, VALID window matches PyTorch AvgPool2d floor behaviour.
    p = CONFIG['pool_size']
    s = lax.reduce_window(x, jnp.zeros((), x.dtype), lax.add,
                          (1, p, p, 1), (1, p, p, 1), 'VALID')
    return s / (p * p)


def _conv(x, w):
    c = CONFIG
    pad = c['padding']
    d = c['dilation']
    return lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=[(pad, pad), (pad, pad)],
        rhs_dilation=(d, d), dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=c['groups'])


def workload(x, w_h2h, w_h2l):
    x = jnp.transpose(x, (0, 2, 3, 1))
    if CONFIG['stride'] == 2:
        x = _avg_pool(x)
    x_h = _conv(x, w_h2h)
    x_l = _conv(_avg_pool(x), w_h2l)
    return jnp.transpose(x_h, (0, 3, 1, 2)), jnp.transpose(x_l, (0, 3, 1, 2))
