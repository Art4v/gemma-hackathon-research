"""[bf16 variant] Bilinear upsample x2 (align_corners=True) followed by 1x1 Conv2d with bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Upsample_17972',
    'batch': 16,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'scale_factor': 2,
    'kernel_size': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    c = CONFIG
    x = jax.random.uniform(kx, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['out_channels'], c['in_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def _align_corners_coords(n_in, n_out):
    # Matches PyTorch: src = o * (in-1)/(out-1), i1 clamped to the last index.
    scale = (n_in - 1) / (n_out - 1) if n_out > 1 else 0.0
    src = jnp.arange(n_out, dtype=jnp.float32) * jnp.float32(scale)
    i0 = jnp.clip(jnp.floor(src).astype(jnp.int32), 0, n_in - 1)
    i1 = jnp.minimum(i0 + 1, n_in - 1)
    lam = src - i0.astype(jnp.float32)
    return i0, i1, lam


def _upsample_bilinear_nhwc(x, scale_factor):
    _, h, w, _ = x.shape
    h0, h1, lh = _align_corners_coords(h, h * scale_factor)
    w0, w1, lw = _align_corners_coords(w, w * scale_factor)
    lh = lh.astype(x.dtype)[None, :, None, None]
    lw = lw.astype(x.dtype)[None, None, :, None]
    rows = jnp.take(x, h0, axis=1) * (1 - lh) + jnp.take(x, h1, axis=1) * lh
    return jnp.take(rows, w0, axis=2) * (1 - lw) + jnp.take(rows, w1, axis=2) * lw


def workload(x, w, b):
    x = jnp.transpose(x, (0, 2, 3, 1))
    x = _upsample_bilinear_nhwc(x, CONFIG['scale_factor'])
    y = lax.conv_general_dilated(
        x, w,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = y + b[None, None, None, :]
    return jnp.transpose(y, (0, 3, 1, 2))
