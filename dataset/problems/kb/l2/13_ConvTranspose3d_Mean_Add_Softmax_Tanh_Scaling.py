"""ConvTranspose3d -> mean over depth -> bias add -> softmax(channels) -> tanh -> scaling."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '13_ConvTranspose3d_Mean_Add_Softmax_Tanh_Scaling',
    'batch': 16,
    'in_channels': 16,
    'out_channels': 64,
    'depth': 32,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'output_padding': 0,
    'scaling_factor': 2.0,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    key = jax.random.key(0)
    kx, kw, kb, kb2 = jax.random.split(key, 4)
    x = jax.random.uniform(
        kx, (c['batch'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    k = c['kernel_size']
    w = jax.random.normal(kw, (c['in_channels'], c['out_channels'], k, k, k), dtype) * 0.02
    conv_b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    bias = jax.random.normal(kb2, (1, c['out_channels'], 1, 1, 1), dtype) * 0.02
    return x, w, conv_b, bias


def workload(x, w, conv_b, bias):
    c = CONFIG
    k = w.shape[2]
    s = c['stride']
    p = c['padding']
    op = c['output_padding']
    # NCDHW -> NDHWC
    xh = jnp.transpose(x, (0, 2, 3, 4, 1))
    # PyTorch transposed-conv weight (I, O, kd, kh, kw) -> flipped DHWIO
    wf = jnp.flip(w, axis=(2, 3, 4))
    wf = jnp.transpose(wf, (2, 3, 4, 0, 1))
    lo = k - 1 - p
    hi = k - 1 - p + op
    y = lax.conv_general_dilated(
        xh, wf,
        window_strides=(1, 1, 1),
        padding=((lo, hi), (lo, hi), (lo, hi)),
        lhs_dilation=(s, s, s),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    y = y + conv_b
    y = jnp.transpose(y, (0, 4, 1, 2, 3))  # NCDHW
    y = jnp.mean(y, axis=2, keepdims=True)
    y = y + bias
    y = jax.nn.softmax(y, axis=1)
    y = jnp.tanh(y)
    return y * c['scaling_factor']
