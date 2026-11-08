"""ConvTranspose3d -> BatchNorm3d (batch stats) -> AvgPool3d(2) -> AvgPool3d(2)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool',
    'batch': 16,
    'in_channels': 32,
    'out_channels': 16,
    'depth': 32,
    'height': 32,
    'width': 32,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'eps': 1e-5,
    'pool_kernel': 2,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    k = c['kernel_size']
    x = jax.random.uniform(
        kx, (c['batch'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['in_channels'], c['out_channels'], k, k, k), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    gamma = jnp.ones((c['out_channels'],), dtype)
    beta = jnp.zeros((c['out_channels'],), dtype)
    return x, w, b, gamma, beta


def _avg_pool(x, p):
    s = lax.reduce_window(x, jnp.array(0, x.dtype), lax.add,
                          (1, p, p, p, 1), (1, p, p, p, 1), 'VALID')
    return s / (p * p * p)


def workload(x, w, b, gamma, beta):
    c = CONFIG
    s = c['stride']
    p = c['padding']
    k = w.shape[2]
    x = jnp.transpose(x, (0, 2, 3, 4, 1))  # NDHWC
    # PyTorch (Cin, Cout, kD, kH, kW) -> flipped DHWIO
    wk = jnp.flip(w, axis=(2, 3, 4))
    wk = jnp.transpose(wk, (2, 3, 4, 0, 1))
    pad = k - 1 - p
    y = lax.conv_general_dilated(
        x, wk,
        window_strides=(1, 1, 1),
        padding=((pad, pad), (pad, pad), (pad, pad)),
        lhs_dilation=(s, s, s),
        rhs_dilation=(1, 1, 1),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'))
    y = y + b
    mean = jnp.mean(y, axis=(0, 1, 2, 3), keepdims=True)
    var = jnp.mean(jnp.square(y - mean), axis=(0, 1, 2, 3), keepdims=True)
    y = (y - mean) * lax.rsqrt(var + c['eps']) * gamma + beta
    y = _avg_pool(y, c['pool_kernel'])
    y = _avg_pool(y, c['pool_kernel'])
    return jnp.transpose(y, (0, 4, 1, 2, 3))
