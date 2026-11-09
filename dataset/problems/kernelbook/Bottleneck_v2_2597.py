"""[bf16 variant] Bottleneck_v2: conv1x1 -> conv1x1+relu -> conv3x3+relu -> conv1x1 -> residual add -> relu."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Bottleneck_v2_2597',
    'batch': 16,
    'in_channels': 64,
    'expand_channels': 256,
    'mid_channels': 64,
    'height': 256,
    'width': 256,
    'kernel_mid': 3,
    'padding_mid': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    k = c['kernel_mid']
    w0 = jax.random.normal(keys[1], (1, 1, c['in_channels'], c['expand_channels']), dtype) * 0.02
    b0 = jax.random.normal(keys[2], (c['expand_channels'],), dtype) * 0.02
    w1 = jax.random.normal(keys[3], (1, 1, c['expand_channels'], c['mid_channels']), dtype) * 0.02
    b1 = jax.random.normal(keys[4], (c['mid_channels'],), dtype) * 0.02
    w2 = jax.random.normal(keys[5], (k, k, c['mid_channels'], c['mid_channels']), dtype) * 0.02
    b2 = jax.random.normal(keys[6], (c['mid_channels'],), dtype) * 0.02
    w3 = jax.random.normal(keys[7], (1, 1, c['mid_channels'], c['expand_channels']), dtype) * 0.02
    b3 = jax.random.normal(keys[8], (c['expand_channels'],), dtype) * 0.02
    return x, w0, b0, w1, b1, w2, b2, w3, b3


def _conv(x, w, b, pad):
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    return y + b


def workload(x, w0, b0, w1, b1, w2, b2, w3, b3):
    pad = CONFIG['padding_mid']
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _conv(h, w0, b0, 0)
    y = jax.nn.relu(_conv(h, w1, b1, 0))
    y = jax.nn.relu(_conv(y, w2, b2, pad))
    y = _conv(y, w3, b3, 0)
    out = jax.nn.relu(y + h)
    return jnp.transpose(out, (0, 3, 1, 2))
