"""ShakeResNet(depth=1): Conv2d(3->16, 3x3, bias) -> ReLU -> AvgPool(8) -> view(-1, 16) -> Linear(16->4)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ShakeResNet_2438',
    'batch': 32,
    'in_channels': 3,
    'height': 512,
    'width': 512,
    'stem_channels': 16,
    'kernel_size': 3,
    'padding': 1,
    'pool': 8,
    'fc_in': 16,
    'labels': 4,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(k[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    conv_w = jax.random.normal(
        k[1], (c['kernel_size'], c['kernel_size'], c['in_channels'], c['stem_channels']), dtype) * 0.02
    conv_b = jax.random.normal(k[2], (c['stem_channels'],), dtype) * 0.02
    fc_w = jax.random.normal(k[3], (c['fc_in'], c['labels']), dtype) * 0.02
    fc_b = jax.random.normal(k[4], (c['labels'],), dtype) * 0.02
    return x, conv_w, conv_b, fc_w, fc_b


def workload(x, conv_w, conv_b, fc_w, fc_b):
    p = CONFIG['padding']
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = lax.conv_general_dilated(
        h, conv_w, window_strides=(1, 1), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST) + conv_b
    h = jax.nn.relu(h)
    k = CONFIG['pool']
    n, hh, ww, ch = h.shape
    h = h.reshape(n, hh // k, k, ww // k, k, ch).mean(axis=(2, 4))
    # Back to NCHW so the flat reshape matches torch's h.view(-1, in_chs[3]).
    h = jnp.transpose(h, (0, 3, 1, 2)).reshape(-1, fc_w.shape[0])
    return jnp.dot(h, fc_w, precision=lax.Precision.HIGHEST) + fc_b
