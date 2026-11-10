"""PreActBlockNoBN: relu -> conv3x3 -> relu -> conv3x3 -> + identity shortcut."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'PreActBlockNoBN_5846',
    'batch': 16,
    'in_planes': 128,
    'planes': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = c['kernel_size']
    kx, k1, kb1, k2, kb2 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (c['batch'], c['in_planes'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(k1, (k, k, c['in_planes'], c['planes']), dtype) * 0.02
    b1 = jax.random.normal(kb1, (c['planes'],), dtype) * 0.02
    w2 = jax.random.normal(k2, (k, k, c['planes'], c['planes']), dtype) * 0.02
    b2 = jax.random.normal(kb2, (c['planes'],), dtype) * 0.02
    return x, w1, b1, w2, b2


def _conv(x_nhwc, w_hwio, b, stride):
    p = CONFIG['padding']
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(stride, stride),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    return y + b


def workload(x, w1, b1, w2, b2):
    # ponytail: stride=1 and in_planes == planes, so the shortcut is the identity on pre-relu x.
    h = jnp.transpose(x, (0, 2, 3, 1))
    out = _conv(jax.nn.relu(h), w1, b1, CONFIG['stride'])
    out = _conv(jax.nn.relu(out), w2, b2, 1)
    out = out + h
    return jnp.transpose(out, (0, 3, 1, 2))
