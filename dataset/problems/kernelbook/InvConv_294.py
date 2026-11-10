"""Glow invertible 1x1 conv: z = conv1x1(x, W), sldj + slogdet(W) * H * W."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'InvConv_294',
    'batch': 8,
    'num_channels': 256,
    'height': 128,
    'width': 128,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_s, k_w = jax.random.split(jax.random.key(0), 3)
    n, c = CONFIG['batch'], CONFIG['num_channels']
    h, w = CONFIG['height'], CONFIG['width']
    x = jax.random.uniform(k_x, (n, c, h, w), dtype)
    sldj = jax.random.uniform(k_s, (n, c, h, w), dtype)
    # Orthogonal init via QR, matching the module's w_init.
    weight = jnp.linalg.qr(jax.random.normal(k_w, (c, c), jnp.float32))[0].astype(dtype)
    return x, sldj, weight


def workload(x, sldj, weight):
    h, w = x.shape[2], x.shape[3]
    ldj = jnp.linalg.slogdet(weight)[1] * h * w
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    kernel = weight.T[None, None, :, :]  # (out, in) -> HWIO
    z = lax.conv_general_dilated(
        x_nhwc, kernel, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    z = jnp.transpose(z, (0, 3, 1, 2))
    return z, sldj + ldj.astype(sldj.dtype)
