"""ConvTranspose2d -> Softmax(dim=1) -> BiasAdd -> Scaling -> Sigmoid."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '91_ConvTranspose2d_Softmax_BiasAdd_Scaling_Sigmoid',
    'batch_size': 128,
    'in_channels': 64,
    'out_channels': 128,
    'height': 64,
    'width': 64,
    'kernel_size': 4,
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
    'scaling_factor': 2.0,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k0, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    x = jax.random.uniform(k0, (c['batch_size'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(k1, (c['in_channels'], c['out_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    conv_b = jax.random.normal(k2, (c['out_channels'],), dtype) * 0.02
    bias = jax.random.normal(k3, (c['out_channels'], 1, 1), dtype) * 0.02
    return x, w, conv_b, bias


def workload(x, w, conv_b, bias):
    c = CONFIG
    k = w.shape[2]
    s = c['stride']
    p = c['padding']
    op = c['output_padding']
    # PyTorch (in, out, kH, kW) -> flipped HWIO
    kern = jnp.transpose(jnp.flip(w, axis=(2, 3)), (2, 3, 0, 1))
    xh = jnp.transpose(x, (0, 2, 3, 1))
    pad = (k - 1 - p, k - 1 - p + op)
    y = lax.conv_general_dilated(
        xh, kern,
        window_strides=(1, 1),
        padding=(pad, pad),
        lhs_dilation=(s, s),
        rhs_dilation=(1, 1),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + conv_b
    y = jnp.transpose(y, (0, 3, 1, 2))
    y = jax.nn.softmax(y, axis=1)
    y = y + bias
    y = y * c['scaling_factor']
    return jax.nn.sigmoid(y)
