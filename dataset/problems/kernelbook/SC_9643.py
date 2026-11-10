"""[bf16 variant] Channel max + mean concat -> Conv2d(2, 1, 3, padding=1) -> sigmoid -> multiply input by spatial attention scale."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SC_9643',
    'batch': 8,
    'channels': 128,
    'height': 256,
    'width': 256,
    'conv_in_channels': 2,
    'conv_out_channels': 1,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w = jax.random.normal(
        k_w,
        (
            CONFIG['conv_out_channels'],
            CONFIG['conv_in_channels'],
            CONFIG['kernel_size'],
            CONFIG['kernel_size'],
        ),
        dtype,
    ) * 0.02
    b = jax.random.normal(k_b, (CONFIG['conv_out_channels'],), dtype) * 0.02
    return x, w, b


def workload(x, w, b):
    x_compress = jnp.concatenate(
        (jnp.max(x, axis=1, keepdims=True), jnp.mean(x, axis=1, keepdims=True)),
        axis=1,
    )
    nhwc = jnp.transpose(x_compress, (0, 2, 3, 1))
    pad = CONFIG['padding']
    stride = CONFIG['stride']
    out = lax.conv_general_dilated(
        nhwc,
        w,
        window_strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    out = out + b
    x_out = jnp.transpose(out, (0, 3, 1, 2))
    scale = jax.nn.sigmoid(x_out)
    return x * scale
