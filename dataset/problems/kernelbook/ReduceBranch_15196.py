"""ReduceBranch: stride-2 subsample -> 1x1 conv, and shifted (crop+pad) stride-2 subsample -> 1x1 conv, concat on channels."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ReduceBranch_15196',
    'batch': 16,
    'planes': 256,
    'height': 128,
    'width': 128,
    'stride': 2,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w1, k_w2 = jax.random.split(jax.random.key(0), 3)
    b, c, h, w = CONFIG['batch'], CONFIG['planes'], CONFIG['height'], CONFIG['width']
    x = jax.random.uniform(k_x, (b, c, h, w), dtype)
    w1 = jax.random.normal(k_w1, (1, 1, c, c), dtype) * 0.02  # HWIO
    w2 = jax.random.normal(k_w2, (1, 1, c, c), dtype) * 0.02  # HWIO
    return x, w1, w2


def _conv1x1_nhwc(x_nhwc, w_hwio):
    return lax.conv_general_dilated(
        x_nhwc, w_hwio, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))


def workload(x, w1, w2):
    s = CONFIG['stride']
    # ponytail: AvgPool2d(kernel_size=1, stride=s) is exactly a strided slice.
    pooled1 = x[:, :, ::s, ::s]
    shift_x = jnp.pad(x[:, :, 1:, 1:], ((0, 0), (0, 0), (0, 1), (0, 1)))
    pooled2 = shift_x[:, :, ::s, ::s]
    out1 = _conv1x1_nhwc(jnp.transpose(pooled1, (0, 2, 3, 1)), w1)
    out2 = _conv1x1_nhwc(jnp.transpose(pooled2, (0, 2, 3, 1)), w2)
    out = jnp.concatenate([out1, out2], axis=-1)
    return jnp.transpose(out, (0, 3, 1, 2))
