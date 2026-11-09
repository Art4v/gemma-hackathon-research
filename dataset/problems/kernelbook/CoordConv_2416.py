"""[bf16 variant] CoordConv: AddCoords (xx, yy channels in [-1, 1]) -> Conv2d with bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'CoordConv_2416',
    'batch': 8,
    'in_channels': 128,
    'height': 256,
    'width': 256,
    'out_channels': 128,
    'kernel_size': 4,
    'stride': 1,
    'padding': 0,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    ks = c['kernel_size']
    w = jax.random.normal(
        k_w, (ks, ks, c['in_channels'] + 2, c['out_channels']), dtype) * 0.02
    b = jax.random.normal(k_b, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    bsz, _, h, wd = x.shape
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    # Matches torch: xx varies along H (dim 2), yy along W (dim 3), computed in f32.
    xx = (jnp.arange(h, dtype=jnp.float32) / (h - 1)) * 2 - 1
    yy = (jnp.arange(wd, dtype=jnp.float32) / (wd - 1)) * 2 - 1
    xx = jnp.broadcast_to(xx[None, :, None, None], (bsz, h, wd, 1)).astype(x.dtype)
    yy = jnp.broadcast_to(yy[None, None, :, None], (bsz, h, wd, 1)).astype(x.dtype)
    inp = jnp.concatenate([x_nhwc, xx, yy], axis=-1)
    s, p = CONFIG['stride'], CONFIG['padding']
    out = lax.conv_general_dilated(
        inp, w,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    out = out + b
    return jnp.transpose(out, (0, 3, 1, 2))
