"""Policy-value net: conv3x3-relu x3 -> (1x1 conv-relu -> fc -> log_softmax, 1x1 conv-relu -> fc-relu -> fc -> tanh)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Net_4318',
    'batch': 4096,
    'in_channels': 4,
    'board_height': 32,
    'board_width': 32,
    'c1': 32,
    'c2': 64,
    'c3': 128,
    'act_channels': 4,
    'val_channels': 2,
    'val_hidden': 64,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    hw = c['board_height'] * c['board_width']
    keys = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['board_height'], c['board_width']), dtype)

    def w(k, shape):
        return jax.random.normal(k, shape, dtype) * 0.02

    def b(n):
        return jnp.zeros((n,), dtype)

    return (
        x,
        w(keys[1], (c['c1'], c['in_channels'], 3, 3)), b(c['c1']),
        w(keys[2], (c['c2'], c['c1'], 3, 3)), b(c['c2']),
        w(keys[3], (c['c3'], c['c2'], 3, 3)), b(c['c3']),
        w(keys[4], (c['act_channels'], c['c3'], 1, 1)), b(c['act_channels']),
        w(keys[5], (c['val_channels'], c['c3'], 1, 1)), b(c['val_channels']),
        w(keys[6], (c['act_channels'] * hw, hw)), b(hw),
        w(keys[7], (c['val_channels'] * hw, c['val_hidden'])), b(c['val_hidden']),
        w(keys[8], (c['val_hidden'], 1)), b(1),
    )


def _conv(x, w, bias):
    # x NHWC, w OIHW (PyTorch layout); 'SAME' equals padding=k//2 for odd kernels.
    pad = w.shape[2] // 2
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'))
    return y + bias


def _flatten_nchw(x):
    return jnp.transpose(x, (0, 3, 1, 2)).reshape(x.shape[0], -1)


def workload(x, c1w, c1b, c2w, c2b, c3w, c3b, aw, ab, vw, vb,
             afc_w, afc_b, vfc1_w, vfc1_b, vfc2_w, vfc2_b):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = jax.nn.relu(_conv(h, c1w, c1b))
    h = jax.nn.relu(_conv(h, c2w, c2b))
    h = jax.nn.relu(_conv(h, c3w, c3b))

    a = _flatten_nchw(jax.nn.relu(_conv(h, aw, ab)))
    a = jax.nn.log_softmax(a @ afc_w + afc_b, axis=1)

    v = _flatten_nchw(jax.nn.relu(_conv(h, vw, vb)))
    v = jax.nn.relu(v @ vfc1_w + vfc1_b)
    v = jnp.tanh(v @ vfc2_w + vfc2_b)
    return a, v
