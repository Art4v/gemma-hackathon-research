"""[bf16 variant] ConvTranspose3d -> Swish -> GroupNorm -> HardSwish."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '60_ConvTranspose3d_Swish_GroupNorm_HardSwish',
    'batch': 16,
    'in_channels': 64,
    'out_channels': 16,
    'depth': 16,
    'height': 32,
    'width': 32,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'groups': 4,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k = jax.random.key(0)
    kx, kw, kb = jax.random.split(k, 3)
    ks = c['kernel_size']
    x = jax.random.uniform(kx, (c['batch'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['in_channels'], c['out_channels'], ks, ks, ks), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    gamma = jnp.ones((c['out_channels'],), dtype)
    beta = jnp.zeros((c['out_channels'],), dtype)
    return x, w, b, gamma, beta


def workload(x, w, b, gamma, beta):
    c = CONFIG
    s = c['stride']
    p = c['padding']
    ks = w.shape[2]
    pad = ks - 1 - p
    # NCDHW -> NDHWC
    xh = jnp.transpose(x, (0, 2, 3, 4, 1))
    # (I, O, kD, kH, kW) -> flipped DHWIO
    wk = jnp.flip(w, axis=(2, 3, 4))
    wk = jnp.transpose(wk, (2, 3, 4, 0, 1))
    y = lax.conv_general_dilated(
        xh, wk,
        window_strides=(1, 1, 1),
        padding=[(pad, pad)] * 3,
        lhs_dilation=(s, s, s),
        rhs_dilation=(1, 1, 1),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = y + b
    y = jnp.transpose(y, (0, 4, 1, 2, 3))  # back to NCDHW
    # Swish
    y = y * jax.nn.sigmoid(y)
    # GroupNorm
    n, C, D, H, W = y.shape
    g = c['groups']
    yg = y.reshape(n, g, C // g, D, H, W)
    mean = jnp.mean(yg, axis=(2, 3, 4, 5), keepdims=True)
    var = jnp.mean(jnp.square(yg - mean), axis=(2, 3, 4, 5), keepdims=True)
    yg = (yg - mean) * lax.rsqrt(var + c['eps'])
    y = yg.reshape(n, C, D, H, W)
    y = y * gamma.reshape(1, C, 1, 1, 1) + beta.reshape(1, C, 1, 1, 1)
    # HardSwish
    y = y * jnp.clip(y + 3.0, 0.0, 6.0) / 6.0
    return y
