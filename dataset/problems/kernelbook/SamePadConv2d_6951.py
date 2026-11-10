"""SamePadConv2d: TF-style 'same' padding (extra pad bottom/right when odd) followed by Conv2d with bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SamePadConv2d_6951',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 1,
    'dilation': 1,
    'groups': 1,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(k_x, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(
        k_w,
        (c['out_channels'], c['in_channels'] // c['groups'], c['kernel_size'], c['kernel_size']),
        dtype,
    ) * 0.02
    b = jax.random.normal(k_b, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def _same_pad(in_, k, stride, dilation):
    # PyTorch module: pad p//2 both sides, plus 1 at the end if p is odd -> (p//2, p - p//2).
    out = (in_ + stride - 1) // stride
    p = max(0, (out - 1) * stride + (k - 1) * dilation + 1 - in_)
    return (p // 2, p - p // 2)


def workload(x, w, b):
    stride, dilation = CONFIG['stride'], CONFIG['dilation']
    pad_h = _same_pad(x.shape[2], w.shape[2], stride, dilation)
    pad_w = _same_pad(x.shape[3], w.shape[3], stride, dilation)
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(w, (2, 3, 1, 0))
    y = lax.conv_general_dilated(
        x_nhwc,
        w_hwio,
        window_strides=(stride, stride),
        padding=(pad_h, pad_w),
        rhs_dilation=(dilation, dilation),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=CONFIG['groups'],
        precision=lax.Precision.HIGHEST,
    )
    y = y + b
    return jnp.transpose(y, (0, 3, 1, 2))
