"""Conv2dSameExport: TF-style SAME zero-pad (asymmetric) -> Conv2d + bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Conv2dSameExport_16901',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 4,
    'stride': 1,
    'dilation': 1,
}


def _calc_same_pad(size, kernel, stride, dilation):
    total = max((-(size // -stride) - 1) * stride + (kernel - 1) * dilation + 1 - size, 0)
    return (total // 2, total - total // 2)


def create_inputs(dtype=jnp.float32):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    c = CONFIG
    x = jax.random.uniform(k1, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(
        k2, (c['kernel_size'], c['kernel_size'], c['in_channels'], c['out_channels']), dtype) * 0.02
    b = jax.random.normal(k3, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    stride = CONFIG['stride']
    dilation = CONFIG['dilation']
    kh, kw = w.shape[0], w.shape[1]
    h, wd = x.shape[2], x.shape[3]
    padding = [
        _calc_same_pad(h, kh, stride, dilation),
        _calc_same_pad(wd, kw, stride, dilation),
    ]
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x_nhwc, w,
        window_strides=(stride, stride),
        padding=padding,
        rhs_dilation=(dilation, dilation),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + b
    return jnp.transpose(y, (0, 3, 1, 2))
