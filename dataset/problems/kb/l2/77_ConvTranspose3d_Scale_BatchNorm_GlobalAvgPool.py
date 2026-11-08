"""ConvTranspose3d -> scale -> BatchNorm3d (batch stats) -> global average pool."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool',
    'batch': 16,
    'in_channels': 64,
    'out_channels': 128,
    'depth': 16,
    'height': 32,
    'width': 32,
    'kernel_size': 5,
    'scale_factor': 2.0,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x, (c['batch'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    k = c['kernel_size']
    # PyTorch ConvTranspose3d weight layout: (in, out, kD, kH, kW)
    w = jax.random.normal(k_w, (c['in_channels'], c['out_channels'], k, k, k), dtype) * 0.02
    b = jax.random.normal(k_b, (c['out_channels'],), dtype) * 0.02
    gamma = jnp.ones((c['out_channels'],), dtype)
    beta = jnp.zeros((c['out_channels'],), dtype)
    return x, w, b, gamma, beta


def workload(x, w, b, gamma, beta):
    kd, kh, kw = w.shape[2], w.shape[3], w.shape[4]
    x_nhwc = jnp.transpose(x, (0, 2, 3, 4, 1))  # NDHWC
    # (I, O, kD, kH, kW) -> (kD, kH, kW, I, O), spatially flipped for stride-1 transpose conv
    w_dhwio = jnp.transpose(w, (2, 3, 4, 0, 1))[::-1, ::-1, ::-1, :, :]
    y = lax.conv_general_dilated(
        x_nhwc, w_dhwio,
        window_strides=(1, 1, 1),
        padding=((kd - 1, kd - 1), (kh - 1, kh - 1), (kw - 1, kw - 1)),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = y + b
    y = y * CONFIG['scale_factor']
    mean = jnp.mean(y, axis=(0, 1, 2, 3), keepdims=True)
    var = jnp.mean(jnp.square(y - mean), axis=(0, 1, 2, 3), keepdims=True)
    y = (y - mean) * lax.rsqrt(var + CONFIG['eps']) * gamma + beta
    pooled = jnp.mean(y, axis=(1, 2, 3), keepdims=True)  # (N, 1, 1, 1, C)
    return jnp.transpose(pooled, (0, 4, 1, 2, 3))  # (N, C, 1, 1, 1)
