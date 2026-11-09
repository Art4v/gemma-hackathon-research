"""[bf16 variant] ConvTranspose3d -> scale1 -> AvgPool3d(2) -> bias add -> scale2."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '50_ConvTranspose3d_Scaling_AvgPool_BiasAdd_Scaling',
    'batch': 128,
    'in_channels': 8,
    'out_channels': 16,
    'depth': 16,
    'height': 32,
    'width': 32,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'pool_kernel': 2,
    'scale1': 0.5,
    'scale2': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    key = jax.random.key(0)
    kx, kw, kb, kbias = jax.random.split(key, 4)
    k = c['kernel_size']
    x = jax.random.uniform(
        kx, (c['batch'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['in_channels'], c['out_channels'], k, k, k), dtype) * 0.02
    conv_b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    scale1 = jnp.array(c['scale1'], dtype)
    bias = jax.random.normal(kbias, (c['out_channels'], 1, 1, 1), dtype) * 0.02
    scale2 = jnp.array(c['scale2'], dtype)
    return x, w, conv_b, scale1, bias, scale2


def workload(x, w, conv_b, scale1, bias, scale2):
    c = CONFIG
    k = w.shape[2]
    s = c['stride']
    p = c['padding']
    pad = k - 1 - p
    # NCDHW -> NDHWC
    xh = jnp.transpose(x, (0, 2, 3, 4, 1))
    # torch ConvTranspose weight (Cin, Cout, kD, kH, kW) -> flipped DHWIO
    wk = jnp.transpose(w[:, :, ::-1, ::-1, ::-1], (2, 3, 4, 0, 1))
    y = lax.conv_general_dilated(
        xh, wk,
        window_strides=(1, 1, 1),
        padding=((pad, pad), (pad, pad), (pad, pad)),
        lhs_dilation=(s, s, s),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    y = y + conv_b
    y = y * scale1
    pk = c['pool_kernel']
    y = lax.reduce_window(
        y, jnp.array(0, y.dtype), lax.add,
        window_dimensions=(1, pk, pk, pk, 1),
        window_strides=(1, pk, pk, pk, 1),
        padding='VALID',
    ) / (pk ** 3)
    # NDHWC -> NCDHW
    y = jnp.transpose(y, (0, 4, 1, 2, 3))
    y = y + bias
    y = y * scale2
    return y
