"""[bf16 variant] SpatialGate: ConvTranspose2d(C->1, k=3, s=1, p=1) followed by sigmoid."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SpatialGate_8630',
    'batch': 16,
    'in_channels': 128,
    'out_channels': 1,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    c = CONFIG
    x = jax.random.uniform(k_x, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    # PyTorch ConvTranspose2d weight layout: (in, out, kH, kW).
    w = jax.random.normal(k_w, (c['in_channels'], c['out_channels'], c['kernel_size'], c['kernel_size']), dtype) * 0.02
    b = jax.random.normal(k_b, (c['out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    k = w.shape[2]
    pad = k - 1 - CONFIG['padding']
    # Stride-1 transposed conv == cross-correlation with spatially flipped kernel, in/out swapped.
    w_hwio = jnp.transpose(w[:, :, ::-1, ::-1], (2, 3, 0, 1))
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(CONFIG['stride'], CONFIG['stride']),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = y + b
    return jax.nn.sigmoid(jnp.transpose(y, (0, 3, 1, 2)))
