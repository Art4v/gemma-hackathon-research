"""Weight-standardized Conv3d: per-(out, kw) var_mean standardization of weight over dims [1,2,3], then conv3d + bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'StdConv3d_9599',
    'batch': 8,
    'in_channels': 16,
    'out_channels': 64,
    'depth': 64,
    'height': 64,
    'width': 64,
    'kernel_size': 4,
    'stride': 1,
    'padding': 0,
    'dilation': 1,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    c = CONFIG
    k = c['kernel_size']
    x = jax.random.uniform(
        k_x, (c['batch'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    w = jax.random.normal(k_w, (c['out_channels'], c['in_channels'], k, k, k), dtype) * 0.02
    b = jax.random.normal(k_b, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    # PyTorch reduces over dims [1,2,3] of the 5D weight (in, kd, kh), not kw.
    m = jnp.mean(w, axis=(1, 2, 3), keepdims=True)
    v = jnp.var(w, axis=(1, 2, 3), keepdims=True)
    w = (w - m) / jnp.sqrt(v + CONFIG['eps'])

    unbatched = x.ndim == 4  # torch conv3d accepts unbatched (C, D, H, W)
    if unbatched:
        x = x[None]
    x = jnp.transpose(x, (0, 2, 3, 4, 1))      # NCDHW -> NDHWC
    w = jnp.transpose(w, (2, 3, 4, 1, 0))      # OIDHW -> DHWIO
    s, p, d = CONFIG['stride'], CONFIG['padding'], CONFIG['dilation']
    y = lax.conv_general_dilated(
        x, w,
        window_strides=(s, s, s),
        padding=((p, p), (p, p), (p, p)),
        rhs_dilation=(d, d, d),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = y + b
    y = jnp.transpose(y, (0, 4, 1, 2, 3))      # NDHWC -> NCDHW
    if unbatched:
        y = y[0]
    return y
