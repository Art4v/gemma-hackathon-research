"""CoordConv: append normalized (xx, yy) coordinate channels, then Conv2d (VALID, stride 1, bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'CoordConv_1465',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 1,
    'padding': 0,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    b, c, h, w = CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']
    o, k = CONFIG['out_channels'], CONFIG['kernel_size']
    x = jax.random.uniform(k_x, (b, c, h, w), dtype)
    weight = jax.random.normal(k_w, (o, c + 2, k, k), dtype) * 0.02  # OIHW, PyTorch layout
    bias = jax.random.normal(k_b, (o,), dtype) * 0.02
    return x, weight, bias


def workload(x, weight, bias):
    b, _, h, w = x.shape
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    # Matches torch AddCoords: xx varies along H (dim 2), yy along W (dim 3), computed in f32.
    xx = jnp.arange(h, dtype=jnp.float32) / (h - 1) * 2 - 1
    yy = jnp.arange(w, dtype=jnp.float32) / (w - 1) * 2 - 1
    xx = jnp.broadcast_to(xx[None, :, None, None], (b, h, w, 1)).astype(x.dtype)
    yy = jnp.broadcast_to(yy[None, None, :, None], (b, h, w, 1)).astype(x.dtype)
    inp = jnp.concatenate([x_nhwc, xx, yy], axis=-1)
    s, p = CONFIG['stride'], CONFIG['padding']
    out = lax.conv_general_dilated(
        inp, weight,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    out = out + bias[None, None, None, :]
    return jnp.transpose(out, (0, 3, 1, 2))
