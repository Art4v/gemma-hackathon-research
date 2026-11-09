"""[bf16 variant] ConvTranspose2d (k=3, stride=2, padding=0, bias) upsampling block."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'UpBlock_12859',
    'batch': 8,
    'in_f': 128,
    'out_f': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'stride': 2,
    'padding': 0,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['in_f'], CONFIG['height'], CONFIG['width']), dtype)
    k = CONFIG['kernel_size']
    # PyTorch ConvTranspose2d weight layout: (in, out, kh, kw).
    w = jax.random.normal(k_w, (CONFIG['in_f'], CONFIG['out_f'], k, k), dtype) * 0.02
    b = jax.random.normal(k_b, (CONFIG['out_f'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    k = w.shape[2]
    s = CONFIG['stride']
    pad = k - 1 - CONFIG['padding']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    # Transposed conv == regular conv on the stride-dilated input with a flipped kernel.
    w_hwio = jnp.transpose(w[:, :, ::-1, ::-1], (2, 3, 0, 1))
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(1, 1),
        padding=((pad, pad), (pad, pad)),
        lhs_dilation=(s, s),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    y = y + b
    return jnp.transpose(y, (0, 3, 1, 2))
