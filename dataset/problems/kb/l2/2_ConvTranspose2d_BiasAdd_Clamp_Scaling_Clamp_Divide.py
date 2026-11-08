"""ConvTranspose2d -> bias add -> clamp(0,1) -> scale -> clamp(0,1) -> divide by scale."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide',
    'batch': 64,
    'in_channels': 64,
    'out_channels': 64,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
    'scaling_factor': 2.0,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(k1, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(k2, (c['in_channels'], c['out_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    conv_b = jax.random.normal(k3, (c['out_channels'],), dtype) * 0.02
    bias = jax.random.normal(k4, (c['out_channels'], 1, 1), dtype) * 0.02
    return x, w, conv_b, bias


def workload(x, w, conv_b, bias):
    c = CONFIG
    k = w.shape[2]
    s, p, op = c['stride'], c['padding'], c['output_padding']
    scale = c['scaling_factor']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    # (Cin, Cout, kH, kW) -> flipped (kH, kW, Cin, Cout)
    w_hwio = jnp.transpose(jnp.flip(w, axis=(2, 3)), (2, 3, 0, 1))
    lo = k - 1 - p
    hi = k - 1 - p + op
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(1, 1),
        padding=[(lo, hi), (lo, hi)],
        lhs_dilation=(s, s),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + conv_b
    y = jnp.transpose(y, (0, 3, 1, 2))
    y = y + bias
    y = jnp.clip(y, 0.0, 1.0)
    y = y * scale
    y = jnp.clip(y, 0.0, 1.0)
    y = y / scale
    return y
