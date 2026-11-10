"""[bf16 variant] MFM: Conv2d(3x3, Cin -> 2*Cout, pad 1) -> split channels in half -> elementwise max."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'mfm_59',
    'batch': 16,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    ks = c['kernel_size']
    w = jax.random.normal(
        k_w, (ks, ks, c['in_channels'], 2 * c['out_channels']), dtype) * 0.02
    b = jax.random.normal(k_b, (2 * c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    pad = CONFIG['padding']
    stride = CONFIG['stride']
    out_c = w.shape[-1] // 2
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x_nhwc, w,
        window_strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    ) + b
    out = jnp.maximum(y[..., :out_c], y[..., out_c:])
    return jnp.transpose(out, (0, 3, 1, 2))
