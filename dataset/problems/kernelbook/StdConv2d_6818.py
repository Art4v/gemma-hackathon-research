"""StdConv2d: per-filter weight standardization (mean/var over in,kh,kw) followed by conv2d with bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'StdConv2d_6818',
    'batch': 16,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 1,
    'padding': 0,
    'eps': 1e-10,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(k_x, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(
        k_w, (c['out_channels'], c['in_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    b = jax.random.normal(k_b, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    # w is OIHW like PyTorch; standardize per output channel.
    m = jnp.mean(w, axis=(1, 2, 3), keepdims=True)
    v = jnp.var(w, axis=(1, 2, 3), keepdims=True)
    w = (w - m) / jnp.sqrt(v + CONFIG['eps'])
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(w, (2, 3, 1, 0))
    s, p = CONFIG['stride'], CONFIG['padding']
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = y + b
    return jnp.transpose(y, (0, 3, 1, 2))
