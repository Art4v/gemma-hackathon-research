"""[bf16 variant] ConvTranspose3d -> MaxPool3d -> Softmax(channels) -> Subtract -> Swish -> Max(channels)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '89_ConvTranspose3d_MaxPool_Softmax_Subtract_Swish_Max',
    'batch_size': 256,
    'in_channels': 3,
    'out_channels': 16,
    'depth': 16,
    'height': 32,
    'width': 32,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
    'pool_kernel_size': 2,
    'pool_stride': 2,
    'pool_padding': 0,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k_x, k_w, k_b, k_s = jax.random.split(jax.random.key(0), 4)
    k = c['kernel_size']
    x = jax.random.uniform(
        k_x, (c['batch_size'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    w = jax.random.normal(k_w, (c['in_channels'], c['out_channels'], k, k, k), dtype) * 0.02
    b = jax.random.normal(k_b, (c['out_channels'],), dtype) * 0.02
    sub = jax.random.normal(k_s, (c['out_channels'],), dtype)
    return x, w, b, sub


def workload(x, w, b, sub):
    c = CONFIG
    k = c['kernel_size']
    s = c['stride']
    p = c['padding']
    op = c['output_padding']
    # NCDHW -> NDHWC
    xh = jnp.transpose(x, (0, 2, 3, 4, 1))
    # PyTorch transposed-conv weight (Cin, Cout, kD, kH, kW) -> flipped DHWIO
    kern = jnp.flip(w, axis=(2, 3, 4))
    kern = jnp.transpose(kern, (2, 3, 4, 0, 1))
    lo = k - 1 - p
    hi = k - 1 - p + op
    y = lax.conv_general_dilated(
        xh, kern,
        window_strides=(1, 1, 1),
        padding=[(lo, hi)] * 3,
        lhs_dilation=(s, s, s),
        rhs_dilation=(1, 1, 1),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
    )
    y = y + b.reshape(1, 1, 1, 1, -1)
    # MaxPool3d
    pk = c['pool_kernel_size']
    ps = c['pool_stride']
    pp = c['pool_padding']
    y = lax.reduce_window(
        y, -jnp.inf, lax.max,
        window_dimensions=(1, pk, pk, pk, 1),
        window_strides=(1, ps, ps, ps, 1),
        padding=((0, 0), (pp, pp), (pp, pp), (pp, pp), (0, 0)),
    )
    # Softmax over channels (last axis in NDHWC)
    y = jax.nn.softmax(y, axis=-1)
    y = y - sub.reshape(1, 1, 1, 1, -1)
    y = jax.nn.sigmoid(y) * y
    # Max over channels -> (N, D, H, W)
    return jnp.max(y, axis=-1)
