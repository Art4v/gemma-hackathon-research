"""[bf16 variant] ConvTranspose3d (asymmetric input, square kernel, no bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '70_conv_transposed_3D__asymmetric_input__square_kernel',
    'batch_size': 8,
    'in_channels': 128,
    'out_channels': 64,
    'kernel_size': 3,
    'depth': 32,
    'height': 48,
    'width': 64,
    'stride': 1,
    'padding': 0,
    'output_padding': 0,
    'dilation': 1,
    'groups': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    kx, kw = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        kx, (c['batch_size'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    k = c['kernel_size']
    w = jax.random.normal(
        kw, (c['in_channels'], c['out_channels'] // c['groups'], k, k, k), dtype) * 0.02
    return x, w


def workload(x, w):
    c = CONFIG
    s = c['stride']
    p = c['padding']
    op = c['output_padding']
    d = c['dilation']
    g = c['groups']
    cin, coutg, kd, kh, kw = w.shape
    # (in, out/g, kd, kh, kw) -> flipped DHWIO with I=in/g, O=out
    wf = jnp.flip(w, axis=(2, 3, 4))
    wf = wf.reshape(g, cin // g, coutg, kd, kh, kw)
    wf = jnp.transpose(wf, (3, 4, 5, 1, 0, 2)).reshape(kd, kh, kw, cin // g, g * coutg)
    pads = []
    for k in (kd, kh, kw):
        lo = d * (k - 1) - p
        pads.append((lo, lo + op))
    xh = jnp.transpose(x, (0, 2, 3, 4, 1))
    y = lax.conv_general_dilated(
        xh, wf,
        window_strides=(1, 1, 1),
        padding=pads,
        lhs_dilation=(s, s, s),
        rhs_dilation=(d, d, d),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        feature_group_count=g,
        precision=lax.Precision.HIGHEST,
    )
    return jnp.transpose(y, (0, 4, 1, 2, 3))
