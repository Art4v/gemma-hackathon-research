"""ConvTranspose2d (grouped, strided, padded, dilated, asymmetric kernel, no bias)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '75_conv_transposed_2D_asymmetric_input_asymmetric_kernel_strided__grouped____padded____dilated__',
    'batch_size': 16,
    'in_channels': 32,
    'out_channels': 64,
    'kernel_size': (3, 5),
    'height': 128,
    'width': 256,
    'stride': (2, 3),
    'padding': (1, 2),
    'dilation': (2, 1),
    'groups': 4,
}


def create_inputs(dtype=jnp.float32):
    k1, k2 = jax.random.split(jax.random.key(0))
    c = CONFIG
    x = jax.random.uniform(k1, (c['batch_size'], c['in_channels'], c['height'], c['width']), dtype)
    kh, kw = c['kernel_size']
    w = jax.random.normal(k2, (c['in_channels'], c['out_channels'] // c['groups'], kh, kw), dtype) * 0.02
    return x, w


def workload(x, w):
    g = CONFIG['groups']
    sh, sw = CONFIG['stride']
    ph, pw = CONFIG['padding']
    dh, dw = CONFIG['dilation']
    cin, cout_g, kh, kw = w.shape
    cin_g = cin // g
    # (Cin, Cout/g, kh, kw) -> (Cout, Cin/g, kh, kw), flipped spatially
    wt = w.reshape(g, cin_g, cout_g, kh, kw)
    wt = jnp.transpose(wt, (0, 2, 1, 3, 4)).reshape(g * cout_g, cin_g, kh, kw)
    wt = wt[:, :, ::-1, ::-1]
    w_hwio = jnp.transpose(wt, (2, 3, 1, 0))
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    pad_h = dh * (kh - 1) - ph
    pad_w = dw * (kw - 1) - pw
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(1, 1),
        padding=((pad_h, pad_h), (pad_w, pad_w)),
        lhs_dilation=(sh, sw),
        rhs_dilation=(dh, dw),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        feature_group_count=g,
    )
    return jnp.transpose(y, (0, 3, 1, 2))
