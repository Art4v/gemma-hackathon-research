"""ConvTranspose2d (no bias) followed by ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SimpleConvTranspose2dModule_3325',
    'batch': 16,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 1,
    'padding': 0,
    'output_padding': 0,
    'dilation': 1,
    'groups': 1,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(k1, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    # PyTorch conv_transpose weight layout: (Cin, Cout/groups, kH, kW)
    w = jax.random.normal(
        k2, (c['in_channels'], c['out_channels'] // c['groups'], c['kernel_size'], c['kernel_size']), dtype
    ) * 0.02
    return x, w


def workload(x, w):
    s, p, op = CONFIG['stride'], CONFIG['padding'], CONFIG['output_padding']
    d, g = CONFIG['dilation'], CONFIG['groups']
    cin, cout_g, kh, kw = w.shape
    # (Cin, Cout/g, kH, kW) -> HWIO with I = Cin/g, O = g*Cout/g, spatially flipped
    w = w.reshape(g, cin // g, cout_g, kh, kw)
    w = jnp.transpose(w, (3, 4, 1, 0, 2)).reshape(kh, kw, cin // g, g * cout_g)
    w = jnp.flip(w, axis=(0, 1))
    x = jnp.transpose(x, (0, 2, 3, 1))
    pads = [(d * (k - 1) - p, d * (k - 1) - p + op) for k in (kh, kw)]
    y = lax.conv_general_dilated(
        x, w,
        window_strides=(1, 1),
        padding=pads,
        lhs_dilation=(s, s),
        rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=g,
    )
    y = jax.nn.relu(y)
    return jnp.transpose(y, (0, 3, 1, 2))
