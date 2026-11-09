"""ExpandingBlock: ConvTranspose2d(k3, s2, p1, op1) -> InstanceNorm2d -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ExpandingBlock_12283',
    'batch': 8,
    'in_channels': 256,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'output_padding': 1,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(k_x, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    # PyTorch ConvTranspose2d weight layout: (in, out, kH, kW).
    w = jax.random.normal(k_w, (c['in_channels'], c['out_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    b = jax.random.normal(k_b, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    c = CONFIG
    k = w.shape[2]
    s = c['stride']
    lo = k - 1 - c['padding']
    hi = lo + c['output_padding']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    # Transposed conv as a forward conv: flip the kernel spatially, (in, out, kH, kW) -> HWIO.
    w_hwio = jnp.transpose(w[:, :, ::-1, ::-1], (2, 3, 0, 1))
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(1, 1),
        padding=((lo, hi), (lo, hi)),
        lhs_dilation=(s, s),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    y = y + b
    mean = jnp.mean(y, axis=(1, 2), keepdims=True)
    var = jnp.var(y, axis=(1, 2), keepdims=True)
    y = (y - mean) * lax.rsqrt(var + c['eps'])
    y = jnp.maximum(y, 0)
    return jnp.transpose(y, (0, 3, 1, 2))
