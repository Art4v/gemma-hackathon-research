"""[bf16 variant] WRNBottleneck: conv1x1+bias+ReLU -> conv3x3(stride)+bias+ReLU -> conv1x1+bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'WRNBottleneck_17683',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'width_factor': 4,
    'mid_channels': int(round(128 // 4 * 4)),
    'height': 128,
    'width': 128,
    'stride': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    cin, cmid, cout = c['in_channels'], c['mid_channels'], c['out_channels']
    keys = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(keys[0], (c['batch'], cin, c['height'], c['width']), dtype)
    w1 = jax.random.normal(keys[1], (1, 1, cin, cmid), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (cmid,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (3, 3, cmid, cmid), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (cmid,), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (1, 1, cmid, cout), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (cout,), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def _conv(x, w, b, stride):
    pad = w.shape[0] // 2
    y = lax.conv_general_dilated(
        x, w,
        window_strides=(stride, stride),
        padding=[(pad, pad), (pad, pad)],
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    return y + b


def workload(x, w1, b1, w2, b2, w3, b3):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = jax.nn.relu(_conv(h, w1, b1, 1))
    h = jax.nn.relu(_conv(h, w2, b2, CONFIG['stride']))
    h = _conv(h, w3, b3, 1)
    return jnp.transpose(h, (0, 3, 1, 2))
