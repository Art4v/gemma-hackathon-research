"""ConvTranspose3d (stride 2, padding 1, square kernel 3, no bias) on NCDHW input."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '73_conv_transposed_3D_asymmetric_input_square_kernel__strided_padded__grouped',
    'batch_size': 8,
    'in_channels': 32,
    'out_channels': 32,
    'kernel_size': 3,
    'depth': 32,
    'height': 64,
    'width': 128,
    'stride': 2,
    'padding': 1,
    'output_padding': 0,
    # KernelBench passes 'groups' positionally into output_padding, which is never
    # forwarded to nn.ConvTranspose3d, so the effective module has groups=1.
    'groups': 1,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k_x, k_w = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k_x, (c['batch_size'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    k = c['kernel_size']
    w = jax.random.normal(
        k_w, (c['in_channels'], c['out_channels'] // c['groups'], k, k, k), dtype) * 0.02
    return x, w


def workload(x, w):
    c = CONFIG
    g = c['groups']
    s = c['stride']
    p = c['padding']
    op = c['output_padding']
    cin, cout_g, kd, kh, kw = w.shape
    cin_g = cin // g
    # torch weight (in, out/g, kD, kH, kW) -> DHWIO with I = in/g, O = out, spatially flipped
    wf = jnp.flip(w, axis=(2, 3, 4))
    wf = wf.reshape(g, cin_g, cout_g, kd, kh, kw)
    wf = jnp.transpose(wf, (3, 4, 5, 1, 0, 2)).reshape(kd, kh, kw, cin_g, g * cout_g)
    x_ndhwc = jnp.transpose(x, (0, 2, 3, 4, 1))
    pads = [(kk - 1 - p, kk - 1 - p + op) for kk in (kd, kh, kw)]
    y = lax.conv_general_dilated(
        x_ndhwc, wf,
        window_strides=(1, 1, 1),
        padding=pads,
        lhs_dilation=(s, s, s),
        rhs_dilation=(1, 1, 1),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        feature_group_count=g,
    )
    return jnp.transpose(y, (0, 4, 1, 2, 3))
