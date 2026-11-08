"""ConvTranspose3d -> BatchNorm3d (batch stats) -> subtract spatial mean."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '15_ConvTranspose3d_BatchNorm_Subtract',
    'batch_size': 32,
    'in_channels': 32,
    'out_channels': 32,
    'depth': 16,
    'height': 32,
    'width': 32,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    k = c['kernel_size']
    x = jax.random.uniform(
        kx, (c['batch_size'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['in_channels'], c['out_channels'], k, k, k), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    gamma = jnp.ones((c['out_channels'],), dtype)
    beta = jnp.zeros((c['out_channels'],), dtype)
    return x, w, b, gamma, beta


def workload(x, w, b, gamma, beta):
    k = w.shape[2]
    s = CONFIG['stride']
    p = CONFIG['padding']
    pad = k - 1 - p
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    # (Cin, Cout, kd, kh, kw) -> flip spatial -> (kd, kh, kw, Cin, Cout)
    w_dhwio = jnp.transpose(jnp.flip(w, axis=(2, 3, 4)), (2, 3, 4, 0, 1))
    y = lax.conv_general_dilated(
        x_ndhwc, w_dhwio,
        window_strides=(1, 1, 1),
        padding=[(pad, pad)] * 3,
        lhs_dilation=(s, s, s),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    y = y + b
    mean = jnp.mean(y, axis=(0, 1, 2, 3), keepdims=True)
    var = jnp.var(y, axis=(0, 1, 2, 3), keepdims=True)
    y = (y - mean) / jnp.sqrt(var + CONFIG['eps']) * gamma + beta
    y = y - jnp.mean(y, axis=(1, 2, 3), keepdims=True)
    return jnp.transpose(y, (0, 4, 1, 2, 3))
