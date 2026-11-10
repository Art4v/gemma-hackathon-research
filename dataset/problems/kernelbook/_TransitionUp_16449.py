"""[bf16 variant] ConvTranspose2d (padding and output_size derived from skip) -> concat with skip on channels."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': '_TransitionUp_16449',
    'batch': 8,
    'num_features': 128,
    'x_height': 128,
    'x_width': 128,
    'skip_height': 256,
    'skip_width': 256,
    'kernel_size': 3,
    'stride': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_skip, k_w, k_b = jax.random.split(jax.random.key(0), 4)
    n = CONFIG['batch']
    c = CONFIG['num_features']
    ks = CONFIG['kernel_size']
    x = jax.random.uniform(k_x, (n, c, CONFIG['x_height'], CONFIG['x_width']), dtype)
    skip = jax.random.uniform(k_skip, (n, c, CONFIG['skip_height'], CONFIG['skip_width']), dtype)
    w = jax.random.normal(k_w, (c, c, ks, ks), dtype) * 0.02
    b = jax.random.normal(k_b, (c,), dtype) * 0.02
    return x, skip, w, b


def workload(x, skip, w, b):
    k = CONFIG['kernel_size']
    s = CONFIG['stride']
    pads = []
    for d in (2, 3):
        p = ((x.shape[d] - 1) * s - skip.shape[d] + k + 1) // 2
        min_size = (x.shape[d] - 1) * s - 2 * p + k
        output_padding = skip.shape[d] - min_size
        lo = k - 1 - p
        pads.append((lo, lo + output_padding))
    kernel = jnp.flip(w, axis=(2, 3))
    up = lax.conv_general_dilated(
        jnp.transpose(x, (0, 2, 3, 1)),
        kernel,
        window_strides=(1, 1),
        padding=pads,
        lhs_dilation=(s, s),
        dimension_numbers=('NHWC', 'IOHW', 'NHWC'),
    )
    up = jnp.transpose(up, (0, 3, 1, 2))
    up = up + b[None, :, None, None]
    return jnp.concatenate([up, skip], axis=1)
