"""[bf16 variant] TransitionUp: ConvTranspose2d (k=3, stride=2) -> center crop to skip H/W -> concat with skip on channels."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'TransitionUp_1083',
    'batch': 8,
    'in_channels': 128,
    'out_channels': 128,
    'skip_channels': 128,
    'height': 128,
    'width': 128,
    'skip_height': 128,
    'skip_width': 128,
    'kernel_size': 3,
    'stride': 2,
    'padding': 0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_skip, k_w, k_b = jax.random.split(key, 4)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    skip = jax.random.uniform(
        k_skip,
        (CONFIG['batch'], CONFIG['skip_channels'], CONFIG['skip_height'], CONFIG['skip_width']),
        dtype,
    )
    weight = jax.random.normal(
        k_w,
        (CONFIG['in_channels'], CONFIG['out_channels'], CONFIG['kernel_size'], CONFIG['kernel_size']),
        dtype,
    ) * 0.02
    bias = jax.random.normal(k_b, (CONFIG['out_channels'],), dtype) * 0.02
    return x, skip, weight, bias


def workload(x, skip, weight, bias):
    stride = CONFIG['stride']
    pad = CONFIG['kernel_size'] - 1 - CONFIG['padding']
    kernel = jnp.transpose(weight, (2, 3, 0, 1))[::-1, ::-1, :, :]
    out = lax.conv_general_dilated(
        jnp.transpose(x, (0, 2, 3, 1)),
        kernel,
        window_strides=(1, 1),
        padding=[(pad, pad), (pad, pad)],
        lhs_dilation=(stride, stride),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    out = out + bias[None, None, None, :]
    out = jnp.transpose(out, (0, 3, 1, 2))
    max_height, max_width = skip.shape[2], skip.shape[3]
    h, w = out.shape[2], out.shape[3]
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    out = out[:, :, xy2:xy2 + max_height, xy1:xy1 + max_width]
    return jnp.concatenate([out, skip], axis=1)
