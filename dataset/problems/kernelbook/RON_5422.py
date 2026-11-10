"""RON: Conv2d 3x3 (pad 1) on bottom + ConvTranspose2d 2x2 stride 2 on top, summed."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'RON_5422',
    'batch': 8,
    'lat_inC': 128,
    'top_inC': 256,
    'outC': 128,
    'height': 128,
    'width': 128,
    'lat_kernel': 3,
    'lat_padding': 1,
    'top_kernel': 2,
    'top_stride': 2,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = jax.random.split(jax.random.key(0), 6)
    n, h, w = c['batch'], c['height'], c['width']
    s = c['top_stride']
    bottom = jax.random.uniform(k[0], (n, c['lat_inC'], h, w), dtype)
    top = jax.random.uniform(k[1], (n, c['top_inC'], h // s, w // s), dtype)
    lk, tk = c['lat_kernel'], c['top_kernel']
    lat_w = jax.random.normal(k[2], (lk, lk, c['lat_inC'], c['outC']), dtype) * 0.02
    lat_b = jax.random.normal(k[3], (c['outC'],), dtype) * 0.02
    # PyTorch ConvTranspose2d layout: (in, out, kH, kW)
    top_w = jax.random.normal(k[4], (c['top_inC'], c['outC'], tk, tk), dtype) * 0.02
    top_b = jax.random.normal(k[5], (c['outC'],), dtype) * 0.02
    return bottom, top, lat_w, lat_b, top_w, top_b


def workload(bottom, top, lat_w, lat_b, top_w, top_b):
    p = CONFIG['lat_padding']
    xb = jnp.transpose(bottom, (0, 2, 3, 1))
    x = lax.conv_general_dilated(
        xb, lat_w, window_strides=(1, 1), padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    ) + lat_b

    # stride == kernel, no padding: output patches don't overlap, so the
    # transposed conv is exactly a per-pixel matmul followed by depth-to-space.
    xt = jnp.transpose(top, (0, 2, 3, 1))
    n, h, w, _ = xt.shape
    kh, kw = top_w.shape[2], top_w.shape[3]
    y = jnp.einsum('nhwc,coab->nhawbo', xt, top_w, precision=lax.Precision.HIGHEST)
    y = y.reshape(n, h * kh, w * kw, top_w.shape[1]) + top_b

    return jnp.transpose(x + y, (0, 3, 1, 2))
