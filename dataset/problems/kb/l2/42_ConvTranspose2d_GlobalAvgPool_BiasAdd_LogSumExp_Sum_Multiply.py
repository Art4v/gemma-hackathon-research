"""ConvTranspose2d -> GlobalAvgPool -> BiasAdd -> LogSumExp(channels) -> Sum(spatial) -> Multiply."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '42_ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply',
    'batch': 16,
    'in_channels': 64,
    'out_channels': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'scale': 10.0,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = jax.random.key(0)
    kx, kw, kb, kb2 = jax.random.split(k, 4)
    x = jax.random.uniform(kx, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['in_channels'], c['out_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    conv_b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    bias = jax.random.normal(kb2, (c['out_channels'], 1, 1), dtype) * 0.02
    return x, w, conv_b, bias


def workload(x, w, conv_b, bias):
    kh, kw = w.shape[2], w.shape[3]
    # Transposed conv (stride 1, pad 0) == conv with full padding and flipped kernel.
    w_hwio = jnp.transpose(jnp.flip(w, axis=(2, 3)), (2, 3, 0, 1))  # (kH, kW, in, out)
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(1, 1),
        padding=((kh - 1, kh - 1), (kw - 1, kw - 1)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + conv_b
    y = jnp.transpose(y, (0, 3, 1, 2))  # NCHW
    y = jnp.mean(y, axis=(2, 3), keepdims=True)  # (B, C, 1, 1)
    y = y + bias
    y = jax.nn.logsumexp(y, axis=1, keepdims=True)  # (B, 1, 1, 1)
    y = jnp.sum(y, axis=(2, 3))  # (B, 1)
    return y * CONFIG['scale']
