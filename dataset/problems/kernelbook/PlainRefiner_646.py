"""[bf16 variant] PlainRefiner: three conv3x3-ReLU blocks, conv3x3 prediction, add raw_alpha, sigmoid."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'PlainRefiner_646',
    'batch': 8,
    'in_channels': 4,
    'alpha_channels': 4,
    'conv_channels': 64,
    'pred_channels': 1,
    'height': 512,
    'width': 512,
    'kernel_size': 3,
    'padding': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 10)
    b = CONFIG['batch']
    h = CONFIG['height']
    w = CONFIG['width']
    c_in = CONFIG['in_channels']
    c_alpha = CONFIG['alpha_channels']
    c = CONFIG['conv_channels']
    c_pred = CONFIG['pred_channels']
    k = CONFIG['kernel_size']
    s = CONFIG['weight_scale']

    x = jax.random.uniform(keys[0], (b, c_in, h, w), dtype)
    raw_alpha = jax.random.uniform(keys[1], (b, c_alpha, h, w), dtype)

    w1 = jax.random.normal(keys[2], (c, c_in, k, k), dtype) * s
    b1 = jax.random.normal(keys[3], (c,), dtype) * s
    w2 = jax.random.normal(keys[4], (c, c, k, k), dtype) * s
    b2 = jax.random.normal(keys[5], (c,), dtype) * s
    w3 = jax.random.normal(keys[6], (c, c, k, k), dtype) * s
    b3 = jax.random.normal(keys[7], (c,), dtype) * s
    w4 = jax.random.normal(keys[8], (c_pred, c, k, k), dtype) * s
    b4 = jax.random.normal(keys[9], (c_pred,), dtype) * s

    return (x, raw_alpha, w1, b1, w2, b2, w3, b3, w4, b4)


def _conv2d(x, w, b):
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x,
        w,
        window_strides=(1, 1),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    return y + b


def workload(x, raw_alpha, w1, b1, w2, b2, w3, b3, w4, b4):
    out = jnp.transpose(x, (0, 2, 3, 1))
    out = jax.nn.relu(_conv2d(out, w1, b1))
    out = jax.nn.relu(_conv2d(out, w2, b2))
    out = jax.nn.relu(_conv2d(out, w3, b3))
    raw_refine = _conv2d(out, w4, b4)
    raw_refine = jnp.transpose(raw_refine, (0, 3, 1, 2))
    return jax.nn.sigmoid(raw_alpha + raw_refine)
