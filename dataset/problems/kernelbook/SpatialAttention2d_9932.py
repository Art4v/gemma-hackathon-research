"""[bf16 variant] Conv2d (1x1, no bias) -> sigmoid -> element-wise multiply with the input (spatial attention)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SpatialAttention2d_9932',
    'batch': 8,
    'channels': 64,
    'height': 256,
    'width': 256,
    'out_channels': 1,
    'kernel_size': 1,
    'stride': 1,
    'padding': 0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w = jax.random.split(key)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w = jax.random.normal(
        k_w,
        (CONFIG['out_channels'], CONFIG['channels'], CONFIG['kernel_size'], CONFIG['kernel_size']),
        dtype,
    ) * 0.02
    return x, w


def workload(x, w):
    stride = CONFIG['stride']
    pad = CONFIG['padding']
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    z = lax.conv_general_dilated(
        x_nhwc,
        w,
        window_strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    z = jax.nn.sigmoid(z)
    return jnp.transpose(x_nhwc * z, (0, 3, 1, 2))
