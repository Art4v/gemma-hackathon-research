"""[bf16 variant] ConvTranspose2d -> MaxPool2d -> Hardtanh -> Mean(H,W) -> Tanh."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh',
    'batch_size': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'output_padding': 0,
    'dilation': 1,
    'maxpool_kernel_size': 2,
    'maxpool_stride': 2,
    'hardtanh_min': -1.0,
    'hardtanh_max': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    x = jax.random.uniform(kx, (c['batch_size'], c['in_channels'], c['height'], c['width']), dtype)
    # PyTorch ConvTranspose2d weight layout: (in, out, kH, kW)
    w = jax.random.normal(kw, (c['in_channels'], c['out_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    c = CONFIG
    k = w.shape[2]
    s = c['stride']
    p = c['padding']
    op = c['output_padding']
    d = c['dilation']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    # (in, out, kH, kW) -> flipped HWIO
    w_hwio = jnp.transpose(w[:, :, ::-1, ::-1], (2, 3, 0, 1))
    lo = d * (k - 1) - p
    hi = d * (k - 1) - p + op
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(1, 1),
        padding=((lo, hi), (lo, hi)),
        lhs_dilation=(s, s),
        rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + b[None, None, None, :]
    mk = c['maxpool_kernel_size']
    ms = c['maxpool_stride']
    y = lax.reduce_window(y, jnp.array(-jnp.inf, y.dtype), lax.max,
                          (1, mk, mk, 1), (1, ms, ms, 1), 'VALID')
    y = jnp.clip(y, c['hardtanh_min'], c['hardtanh_max'])
    y = jnp.mean(y, axis=(1, 2), keepdims=True)
    y = jnp.tanh(y)
    return jnp.transpose(y, (0, 3, 1, 2))
