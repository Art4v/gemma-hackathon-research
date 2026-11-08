"""ConvTranspose2d -> Mish -> add scalar -> Hardtanh(-1,1) -> scale."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling',
    'batch': 16,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
    'add_value': 0.5,
    'scale': 2.0,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    x = jax.random.uniform(kx, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(kw, (c['in_channels'], c['out_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    c = CONFIG
    k = w.shape[2]
    s = c['stride']
    p = c['padding']
    op = c['output_padding']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    # PyTorch (in, out, kH, kW) -> flipped HWIO (kH, kW, in, out)
    w_hwio = jnp.transpose(jnp.flip(w, axis=(2, 3)), (2, 3, 0, 1))
    lo = k - 1 - p
    hi = k - 1 - p + op
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(1, 1),
        padding=((lo, hi), (lo, hi)),
        lhs_dilation=(s, s),
        rhs_dilation=(1, 1),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + b.reshape(1, 1, 1, -1)
    y = y * jnp.tanh(jax.nn.softplus(y))
    y = y + c['add_value']
    y = jnp.clip(y, -1.0, 1.0)
    y = y * c['scale']
    return jnp.transpose(y, (0, 3, 1, 2))
