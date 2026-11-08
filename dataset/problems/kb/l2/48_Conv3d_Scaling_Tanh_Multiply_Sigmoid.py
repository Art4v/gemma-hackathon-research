"""Conv3d -> per-channel scale -> tanh -> per-channel bias multiply -> sigmoid."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '48_Conv3d_Scaling_Tanh_Multiply_Sigmoid',
    'batch': 128,
    'in_channels': 3,
    'out_channels': 16,
    'depth': 16,
    'height': 64,
    'width': 64,
    'kernel_size': 3,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    key = jax.random.key(0)
    kx, kw, kb, ks, kbias = jax.random.split(key, 5)
    x = jax.random.uniform(
        kx, (c['batch'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    k = c['kernel_size']
    w = jax.random.normal(kw, (k, k, k, c['in_channels'], c['out_channels']), dtype) * 0.02
    conv_b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    scale = jax.random.normal(ks, (c['out_channels'],), dtype)
    bias = jax.random.normal(kbias, (c['out_channels'],), dtype)
    return x, w, conv_b, scale, bias


def workload(x, w, conv_b, scale, bias):
    # NCDHW -> NDHWC
    xh = jnp.transpose(x, (0, 2, 3, 4, 1))
    y = lax.conv_general_dilated(
        xh, w,
        window_strides=(1, 1, 1),
        padding='VALID',
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = y + conv_b
    y = y * scale
    y = jnp.tanh(y)
    y = y * bias
    y = jax.nn.sigmoid(y)
    # NDHWC -> NCDHW
    return jnp.transpose(y, (0, 4, 1, 2, 3))
