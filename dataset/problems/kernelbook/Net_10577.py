"""[bf16 variant] Conv3x3x3+ReLU trunk -> policy head (conv1x1, fc, log_softmax) and value head (conv1x1, fc, relu, fc, tanh)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Net_10577',
    'batch': 4096,
    'in_channels': 4,
    'board_height': 32,
    'board_width': 32,
    'conv1_out': 32,
    'conv2_out': 64,
    'conv3_out': 128,
    'act_channels': 4,
    'val_channels': 2,
    'val_hidden': 64,
    'val_out': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    b, h, w = c['batch'], c['board_height'], c['board_width']
    hw = h * w
    keys = jax.random.split(jax.random.key(0), 17)
    x = jax.random.uniform(keys[0], (b, c['in_channels'], h, w), dtype)

    def p(k, shape):
        return jax.random.normal(k, shape, dtype) * 0.02

    return (
        x,
        p(keys[1], (3, 3, c['in_channels'], c['conv1_out'])), p(keys[2], (c['conv1_out'],)),
        p(keys[3], (3, 3, c['conv1_out'], c['conv2_out'])), p(keys[4], (c['conv2_out'],)),
        p(keys[5], (3, 3, c['conv2_out'], c['conv3_out'])), p(keys[6], (c['conv3_out'],)),
        p(keys[7], (1, 1, c['conv3_out'], c['act_channels'])), p(keys[8], (c['act_channels'],)),
        p(keys[9], (c['act_channels'] * hw, hw)), p(keys[10], (hw,)),
        p(keys[11], (1, 1, c['conv3_out'], c['val_channels'])), p(keys[12], (c['val_channels'],)),
        p(keys[13], (c['val_channels'] * hw, c['val_hidden'])), p(keys[14], (c['val_hidden'],)),
        p(keys[15], (c['val_hidden'], c['val_out'])), p(keys[16], (c['val_out'],)),
    )


def _conv(x_nhwc, w_hwio, b):
    kh, kw = w_hwio.shape[0], w_hwio.shape[1]
    pad = ((kh // 2, kh // 2), (kw // 2, kw // 2))
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio, window_strides=(1, 1), padding=pad,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def _flatten_nchw(x_nhwc):
    # Match torch .view on NCHW: channel-major flatten.
    return jnp.transpose(x_nhwc, (0, 3, 1, 2)).reshape(x_nhwc.shape[0], -1)


def workload(x, c1w, c1b, c2w, c2b, c3w, c3b, acw, acb, afw, afb,
             vcw, vcb, vf1w, vf1b, vf2w, vf2b):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = jax.nn.relu(_conv(h, c1w, c1b))
    h = jax.nn.relu(_conv(h, c2w, c2b))
    h = jax.nn.relu(_conv(h, c3w, c3b))

    a = jax.nn.relu(_conv(h, acw, acb))
    a = _flatten_nchw(a)
    x_act = jax.nn.log_softmax(a @ afw + afb, axis=1)

    v = jax.nn.relu(_conv(h, vcw, vcb))
    v = _flatten_nchw(v)
    v = jax.nn.relu(v @ vf1w + vf1b)
    x_val = jnp.tanh(v @ vf2w + vf2b)
    return x_act, x_val
