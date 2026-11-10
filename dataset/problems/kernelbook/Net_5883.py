"""[bf16 variant] Conv3x3 x3 + ReLU trunk -> policy head (conv1x1, ReLU, linear, log_softmax over batch dim) and value head (conv1x1, ReLU, linear, ReLU, linear, tanh)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Net_5883',
    'batch': 4096,
    'in_channels': 4,
    'board_width': 32,
    'board_height': 32,
    'c1': 32,
    'c2': 64,
    'c3': 128,
    'act_channels': 4,
    'val_channels': 2,
    'val_hidden': 64,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    h, w = c['board_height'], c['board_width']
    hw = h * w
    keys = jax.random.split(jax.random.key(0), 17)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], h, w), dtype)
    shapes = [
        (3, 3, c['in_channels'], c['c1']), (c['c1'],),
        (3, 3, c['c1'], c['c2']), (c['c2'],),
        (3, 3, c['c2'], c['c3']), (c['c3'],),
        (1, 1, c['c3'], c['act_channels']), (c['act_channels'],),
        (c['act_channels'] * hw, hw), (hw,),
        (1, 1, c['c3'], c['val_channels']), (c['val_channels'],),
        (c['val_channels'] * hw, c['val_hidden']), (c['val_hidden'],),
        (c['val_hidden'], 1), (1,),
    ]
    params = tuple(jax.random.normal(k, s, dtype) * 0.02 for k, s in zip(keys[1:], shapes))
    return (x,) + params


def _conv(x, w, b):
    y = lax.conv_general_dilated(x, w, (1, 1), 'SAME',
                                 dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def _flatten_nchw(x):
    # torch .view on NCHW flattens channel-major
    return jnp.transpose(x, (0, 3, 1, 2)).reshape(x.shape[0], -1)


def workload(x, w1, b1, w2, b2, w3, b3, wa, ba, wfa, bfa, wv, bv, wfv1, bfv1, wfv2, bfv2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = jax.nn.relu(_conv(h, w1, b1))
    h = jax.nn.relu(_conv(h, w2, b2))
    h = jax.nn.relu(_conv(h, w3, b3))

    a = _flatten_nchw(jax.nn.relu(_conv(h, wa, ba)))
    x_act = jax.nn.log_softmax(a @ wfa + bfa, axis=0)  # dim=0 as in the original

    v = _flatten_nchw(jax.nn.relu(_conv(h, wv, bv)))
    v = jax.nn.relu(v @ wfv1 + bfv1)
    x_val = jnp.tanh(v @ wfv2 + bfv2)
    return x_act, x_val
