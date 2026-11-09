"""ConvTranspose2d(k=2, s=2) upsample of x2 -> pad/crop to x1 spatial size -> concat([up(x2), x1], channel)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'up_9766',
    'batch': 16,
    'in_ch': 128,
    'out_ch': 128,
    'x1_ch': 128,
    'x2_h': 64,
    'x2_w': 64,
    'x1_h': 128,
    'x1_w': 128,
    'kernel_size': 2,
    'stride': 2,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    x1 = jax.random.uniform(k1, (c['batch'], c['x1_ch'], c['x1_h'], c['x1_w']), dtype)
    x2 = jax.random.uniform(k2, (c['batch'], c['in_ch'], c['x2_h'], c['x2_w']), dtype)
    # PyTorch ConvTranspose2d weight layout: (in_ch, out_ch, kH, kW)
    w = jax.random.normal(k3, (c['in_ch'], c['out_ch'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    b = jax.random.normal(k4, (c['out_ch'],), dtype) * 0.02
    return x1, x2, w, b


def workload(x1, x2, w, b):
    s = CONFIG['stride']
    kh, kw = w.shape[2], w.shape[3]
    x = jnp.transpose(x2, (0, 2, 3, 1))
    # Transposed conv == dilated-input conv with a spatially flipped (in, out) kernel.
    kern = jnp.flip(jnp.transpose(w, (2, 3, 0, 1)), axis=(0, 1))
    y = lax.conv_general_dilated(
        x, kern,
        window_strides=(1, 1),
        padding=[(kh - 1, kh - 1), (kw - 1, kw - 1)],
        lhs_dilation=(s, s),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = jnp.transpose(y + b, (0, 3, 1, 2))
    dy = x1.shape[2] - y.shape[2]
    dx = x1.shape[3] - y.shape[3]
    # lax.pad accepts negative padding, matching F.pad's crop behaviour when diff < 0.
    y = lax.pad(
        y, jnp.array(0, y.dtype),
        [(0, 0, 0), (0, 0, 0), (dy // 2, dy - dy // 2, 0), (dx // 2, dx - dx // 2, 0)],
    )
    return jnp.concatenate([y, x1], axis=1)
