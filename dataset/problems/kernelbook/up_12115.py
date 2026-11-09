"""ConvTranspose2d(k=2, s=2) upsample of x2 -> pad to x1 spatial size -> concat [x2_up, x1] on channels."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'up_12115',
    'batch': 8,
    'in_ch': 256,
    'out_ch': 128,
    'x1_ch': 128,
    'x2_h': 64,
    'x2_w': 64,
    'x1_h': 128,
    'x1_w': 128,
    'kernel': 2,
    'stride': 2,
}


def create_inputs(dtype=jnp.float32):
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    c = CONFIG
    x1 = jax.random.uniform(k1, (c['batch'], c['x1_ch'], c['x1_h'], c['x1_w']), dtype)
    x2 = jax.random.uniform(k2, (c['batch'], c['in_ch'], c['x2_h'], c['x2_w']), dtype)
    # PyTorch ConvTranspose2d weight layout: (in, out, kH, kW)
    w = jax.random.normal(k3, (c['in_ch'], c['out_ch'], c['kernel'], c['kernel']), dtype) * 0.02
    b = jax.random.normal(k4, (c['out_ch'],), dtype) * 0.02
    return x1, x2, w, b


def workload(x1, x2, w, b):
    n, _, h, wd = x2.shape
    o, kh, kw = w.shape[1], w.shape[2], w.shape[3]
    # ponytail: kernel == stride, so windows don't overlap and the transposed conv is a per-pixel einsum
    y = jnp.einsum('nchw,coab->nohawb', x2, w).reshape(n, o, h * kh, wd * kw)
    y = y + b[None, :, None, None]
    diff_y = x1.shape[2] - y.shape[2]
    diff_x = x1.shape[3] - y.shape[3]
    # lax.pad accepts negative widths, matching F.pad's cropping when diff < 0
    y = lax.pad(
        y,
        jnp.array(0, y.dtype),
        [(0, 0, 0), (0, 0, 0),
         (diff_y // 2, diff_y - diff_y // 2, 0),
         (diff_x // 2, diff_x - diff_x // 2, 0)],
    )
    return jnp.concatenate([y, x1], axis=1)
