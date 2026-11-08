"""ConvTranspose3d -> Softmax(dim=1) -> Sigmoid."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '49_ConvTranspose3d_Softmax_Sigmoid',
    'batch_size': 16,
    'in_channels': 32,
    'out_channels': 64,
    'D': 16,
    'H': 32,
    'W': 64,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k0, (c['batch_size'], c['in_channels'], c['D'], c['H'], c['W']), dtype)
    k = c['kernel_size']
    w = jax.random.normal(k1, (c['in_channels'], c['out_channels'], k, k, k), dtype) * 0.02
    b = jax.random.normal(k2, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    s = CONFIG['stride']
    p = CONFIG['padding']
    op = CONFIG['output_padding']
    k = w.shape[2]
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    # PyTorch (Cin, Cout, kD, kH, kW) -> flipped DHWIO
    w_dhwio = jnp.transpose(w, (2, 3, 4, 0, 1))[::-1, ::-1, ::-1, :, :]
    pad = [(k - 1 - p, k - 1 - p + op)] * 3
    y = lax.conv_general_dilated(
        x_ndhwc, w_dhwio,
        window_strides=(1, 1, 1),
        padding=pad,
        lhs_dilation=(s, s, s),
        rhs_dilation=(1, 1, 1),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    y = y + b
    y = jax.nn.softmax(y, axis=-1)
    y = jax.nn.sigmoid(y)
    return jnp.transpose(y, (0, 4, 1, 2, 3))
