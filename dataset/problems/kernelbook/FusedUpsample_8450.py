"""FusedUpsample: scaled weight -> 2x2 box-blur of kernel -> conv_transpose2d(stride=2) + bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'FusedUpsample_8450',
    'batch': 8,
    'in_channel': 128,
    'out_channel': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 4,
    'stride': 2,
    'padding': 0,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], CONFIG['in_channel'], CONFIG['height'], CONFIG['width']), dtype)
    weight = jax.random.normal(
        k_w, (CONFIG['in_channel'], CONFIG['out_channel'], CONFIG['kernel_size'], CONFIG['kernel_size']),
        dtype) * 0.02
    bias = jax.random.normal(k_b, (CONFIG['out_channel'],), dtype) * 0.02
    return x, weight, bias


def workload(x, weight, bias):
    in_c, _, k, _ = weight.shape
    multiplier = (2.0 / (in_c * k * k)) ** 0.5
    w = jnp.pad(weight * multiplier, ((0, 0), (0, 0), (1, 1), (1, 1)))
    w = (w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]) / 4
    kb = w.shape[2]
    s = CONFIG['stride']
    p = CONFIG['padding']
    # conv_transpose == lhs-dilated conv with flipped kernel; torch weight is (in, out, kh, kw) -> HWIO
    rhs = jnp.transpose(w[:, :, ::-1, ::-1], (2, 3, 0, 1))
    lhs = jnp.transpose(x, (0, 2, 3, 1))
    pad = kb - 1 - p
    out = lax.conv_general_dilated(
        lhs, rhs,
        window_strides=(1, 1),
        padding=[(pad, pad), (pad, pad)],
        lhs_dilation=(s, s),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    out = out + bias
    return jnp.transpose(out, (0, 3, 1, 2))
