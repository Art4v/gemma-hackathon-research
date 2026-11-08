"""ConvTranspose2d -> add scalar -> min(x, 0) -> GELU -> multiply scalar."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '93_ConvTranspose2d_Add_Min_GELU_Multiply',
    'batch_size': 128,
    'in_channels': 64,
    'out_channels': 128,
    'height': 64,
    'width': 64,
    'kernel_size': 4,
    'stride': 2,
    'padding': 0,
    'add_value': 0.5,
    'multiply_value': 2.0,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw, kb = jax.random.split(key, 3)
    c = CONFIG
    x = jax.random.uniform(
        kx, (c['batch_size'], c['in_channels'], c['height'], c['width']), dtype)
    w = jax.random.normal(
        kw, (c['in_channels'], c['out_channels'], c['kernel_size'], c['kernel_size']),
        dtype) * 0.02
    b = jax.random.normal(kb, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    stride = CONFIG['stride']
    pad = CONFIG['padding']
    kh, kw = w.shape[2], w.shape[3]
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    # PyTorch ConvTranspose weight (in, out, kH, kW) -> flipped HWIO kernel
    k = jnp.transpose(w[:, :, ::-1, ::-1], (2, 3, 0, 1))
    y = lax.conv_general_dilated(
        x_nhwc, k,
        window_strides=(1, 1),
        padding=((kh - 1 - pad, kh - 1 - pad), (kw - 1 - pad, kw - 1 - pad)),
        lhs_dilation=(stride, stride),
        rhs_dilation=(1, 1),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + b
    y = y + CONFIG['add_value']
    y = jnp.minimum(y, jnp.asarray(0.0, y.dtype))
    y = jax.nn.gelu(y, approximate=False)
    y = y * CONFIG['multiply_value']
    return jnp.transpose(y, (0, 3, 1, 2))
