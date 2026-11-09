"""ConvTranspose2d (k=4, s=4, bias) followed by ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Conv2dTransposeBlock_7928',
    'batch': 32,
    'in_dim': 128,
    'out_dim': 128,
    'height': 64,
    'width': 64,
    'ks': 4,
    'stride': 4,
    'padding': 0,
    'output_padding': 0,
}


def create_inputs(dtype=jnp.float32):
    kx, kw, kb = jax.random.split(jax.random.key(0), 3)
    c = CONFIG
    x = jax.random.uniform(kx, (c['batch'], c['in_dim'], c['height'], c['width']), dtype)
    # PyTorch ConvTranspose2d weight layout: (in, out, kH, kW)
    w = jax.random.normal(kw, (c['in_dim'], c['out_dim'], c['ks'], c['ks']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_dim'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    ks, st = CONFIG['ks'], CONFIG['stride']
    p, op = CONFIG['padding'], CONFIG['output_padding']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    # Transposed conv = dilated-input conv with spatially flipped kernel, (in,out,kh,kw) -> HWIO.
    k_hwio = jnp.transpose(w[:, :, ::-1, ::-1], (2, 3, 0, 1))
    pad = (ks - 1 - p, ks - 1 - p + op)
    y = lax.conv_general_dilated(
        x_nhwc, k_hwio,
        window_strides=(1, 1),
        padding=(pad, pad),
        lhs_dilation=(st, st),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = jax.nn.relu(y + b)
    return jnp.transpose(y, (0, 3, 1, 2))
