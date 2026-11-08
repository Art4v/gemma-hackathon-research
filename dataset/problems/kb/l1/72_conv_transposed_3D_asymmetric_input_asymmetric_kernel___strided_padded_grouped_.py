"""ConvTranspose3d (strided, padded, output_padding, grouped, no bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '72_conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_',
    'batch_size': 8,
    'in_channels': 128,
    'out_channels': 128,
    'kernel_size': (3, 5, 7),
    'depth': 8,
    'height': 32,
    'width': 128,
    'stride': (2, 2, 2),
    'padding': (1, 2, 3),
    'output_padding': (1, 1, 1),
    'groups': 4,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    kx, kw = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        kx, (c['batch_size'], c['in_channels'], c['depth'], c['height'], c['width']), dtype)
    w = jax.random.normal(
        kw, (c['in_channels'], c['out_channels'] // c['groups']) + tuple(c['kernel_size']), dtype) * 0.02
    return x, w


def workload(x, w):
    G = CONFIG['groups']
    stride = CONFIG['stride']
    padding = CONFIG['padding']
    out_pad = CONFIG['output_padding']
    cin, cout_g, kd, kh, kw = w.shape
    cin_g = cin // G
    ks = (kd, kh, kw)
    # (Cin, Cout/G, k...) -> (G, Cin/G, Cout/G, k...)
    wk = w.reshape(G, cin_g, cout_g, kd, kh, kw)
    wk = wk[..., ::-1, ::-1, ::-1]
    # -> (kd, kh, kw, Cin/G, G, Cout/G) -> DHWIO with O = G*Cout/G (group-major)
    wk = jnp.transpose(wk, (3, 4, 5, 1, 0, 2)).reshape(kd, kh, kw, cin_g, G * cout_g)
    pads = [(ks[i] - 1 - padding[i], ks[i] - 1 - padding[i] + out_pad[i]) for i in range(3)]
    xh = jnp.transpose(x, (0, 2, 3, 4, 1))
    y = lax.conv_general_dilated(
        xh, wk,
        window_strides=(1, 1, 1),
        padding=pads,
        lhs_dilation=tuple(stride),
        rhs_dilation=(1, 1, 1),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        feature_group_count=G,
    )
    return jnp.transpose(y, (0, 4, 1, 2, 3))
