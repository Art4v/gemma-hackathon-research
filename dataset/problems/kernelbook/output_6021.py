"""[bf16 variant] Three 1x1 conv heads + sigmoid: score, loc*scope, (angle-0.5)*pi, concat -> (score, geo)."""
import math

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'output_6021',
    'batch': 8,
    'in_channels': 32,
    'height': 512,
    'width': 512,
    'score_channels': 1,
    'loc_channels': 4,
    'angle_channels': 1,
    'scope': 512.0,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    c = CONFIG
    cin = c['in_channels']
    x = jax.random.uniform(keys[0], (c['batch'], cin, c['height'], c['width']), dtype)
    w1 = jax.random.normal(keys[1], (1, 1, cin, c['score_channels']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['score_channels'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (1, 1, cin, c['loc_channels']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['loc_channels'],), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (1, 1, cin, c['angle_channels']), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (c['angle_channels'],), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3


def _conv1x1(x_nhwc, w, b):
    y = lax.conv_general_dilated(
        x_nhwc, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def workload(x, w1, b1, w2, b2, w3, b3):
    xh = jnp.transpose(x, (0, 2, 3, 1))
    score = jax.nn.sigmoid(_conv1x1(xh, w1, b1))
    loc = jax.nn.sigmoid(_conv1x1(xh, w2, b2)) * CONFIG['scope']
    angle = (jax.nn.sigmoid(_conv1x1(xh, w3, b3)) - 0.5) * math.pi
    geo = jnp.concatenate((loc, angle), axis=-1)
    return jnp.transpose(score, (0, 3, 1, 2)), jnp.transpose(geo, (0, 3, 1, 2))
