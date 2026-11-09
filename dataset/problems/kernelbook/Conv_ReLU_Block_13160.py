"""Conv_ReLU_Block: dense 3x3 conv-ReLU chain with concats + 1x1 residual branch, add, 1x1 conv, ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Conv_ReLU_Block_13160',
    'batch': 8,
    'channel_in': 128,
    'height': 128,
    'width': 128,
    'branch0_channels': 128,
    'growth': 32,
    'out_channels': 64,
    'kernel_1x1': 1,
    'kernel_3x3': 3,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(keys[0], (c['batch'], c['channel_in'], c['height'], c['width']), dtype)
    k1, k3, g = c['kernel_1x1'], c['kernel_3x3'], c['growth']
    shapes = [
        (k1, k1, c['channel_in'], c['branch0_channels']),
        (k3, k3, c['channel_in'], g),
        (k3, k3, g, g),
        (k3, k3, 2 * g, g),
        (k3, k3, 3 * g, g),
        (k1, k1, c['branch0_channels'], c['out_channels']),
    ]
    ws = tuple(jax.random.normal(k, s, dtype) * 0.02 for k, s in zip(keys[1:], shapes))
    return (x,) + ws


def _conv_relu(x, w):
    # HWIO weights; pad k//2 matches PyTorch padding=0 (1x1) and padding=1 (3x3).
    pad = [(w.shape[0] // 2,) * 2, (w.shape[1] // 2,) * 2]
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=pad,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    return jax.nn.relu(y)


def workload(x, w0, w1, w2, w3, w4, w5):
    h = jnp.transpose(x, (0, 2, 3, 1))
    out_0 = _conv_relu(h, w0)
    out_1 = _conv_relu(h, w1)
    out_2 = _conv_relu(out_1, w2)
    cat_1 = jnp.concatenate((out_1, out_2), axis=-1)
    out_3 = _conv_relu(cat_1, w3)
    cat_2 = jnp.concatenate((cat_1, out_3), axis=-1)
    out_4 = _conv_relu(cat_2, w4)
    cat_3 = jnp.concatenate((cat_2, out_4), axis=-1)
    out = _conv_relu(out_0 + cat_3, w5)
    return jnp.transpose(out, (0, 3, 1, 2))
