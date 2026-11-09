"""[bf16 variant] ConvTranspose3d (asymmetric input, asymmetric kernel (3,5,7), stride 1, no bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '58_conv_transposed_3D__asymmetric_input__asymmetric_kernel',
    'batch_size': 16,
    'in_channels': 32,
    'out_channels': 16,
    'kernel_size': (3, 5, 7),
    'depth_in': 16,
    'height_in': 32,
    'width_in': 128,
    'stride': (1, 1, 1),
    'padding': (0, 0, 0),
    'output_padding': (0, 0, 0),
    'groups': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k1, k2 = jax.random.split(jax.random.key(0))
    x = jax.random.uniform(
        k1, (c['batch_size'], c['in_channels'], c['depth_in'], c['height_in'], c['width_in']), dtype)
    w = jax.random.normal(
        k2, (c['in_channels'], c['out_channels'] // c['groups']) + tuple(c['kernel_size']), dtype) * 0.02
    return x, w


def workload(x, w):
    c = CONFIG
    G = c['groups']
    cin, cout_g = w.shape[0], w.shape[1]
    ks = w.shape[2:]
    # flip spatial dims
    wf = w[:, :, ::-1, ::-1, ::-1]
    # (G, cin/G, cout/G, kD, kH, kW) -> (kD, kH, kW, cin/G, G, cout/G) -> (kD, kH, kW, cin/G, cout)
    wf = wf.reshape((G, cin // G, cout_g) + tuple(ks))
    wf = jnp.transpose(wf, (3, 4, 5, 1, 0, 2))
    wf = wf.reshape(tuple(ks) + (cin // G, G * cout_g))
    pads = [(k - 1 - p, k - 1 - p + op)
            for k, p, op in zip(ks, c['padding'], c['output_padding'])]
    xh = jnp.transpose(x, (0, 2, 3, 4, 1))
    y = lax.conv_general_dilated(
        xh, wf,
        window_strides=(1, 1, 1),
        padding=pads,
        lhs_dilation=tuple(c['stride']),
        rhs_dilation=(1, 1, 1),
        dimension_numbers=('NDHWC', 'DHWIO', 'NDHWC'),
        feature_group_count=G,
    )
    return jnp.transpose(y, (0, 4, 1, 2, 3))
