"""[bf16 variant] SSARDecoder: five chained ConvTranspose2d (k=4, s=2) layers with bias and no activations."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SSARDecoder_17604',
    'batch': 8,
    'height': 128,
    'width': 128,
    'channels': (256, 64, 32, 16, 8, 2),
    'kernel': 4,
    'stride': 2,
    'paddings': ((1, 1), (1, 1), (1, 1), (1, 1), (2, 1)),
}


def create_inputs(dtype=jnp.bfloat16):
    channels = CONFIG['channels']
    kernel = CONFIG['kernel']
    num_layers = len(channels) - 1
    keys = jax.random.split(jax.random.key(0), 1 + 2 * num_layers)
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], channels[0], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    params = []
    for i in range(num_layers):
        w = jax.random.normal(
            keys[1 + 2 * i], (channels[i], channels[i + 1], kernel, kernel), dtype
        ) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (channels[i + 1],), dtype) * 0.02
        params.extend([w, b])
    return (x, *params)


def _deconv(x, w, b, padding):
    kernel = CONFIG['kernel']
    stride = CONFIG['stride']
    y = lax.conv_general_dilated(
        x,
        jnp.flip(w, axis=(2, 3)),
        window_strides=(1, 1),
        padding=[(kernel - 1 - p, kernel - 1 - p) for p in padding],
        lhs_dilation=(stride, stride),
        rhs_dilation=(1, 1),
        dimension_numbers=('NHWC', 'IOHW', 'NHWC'),
    )
    return y + b


def workload(*inputs):
    x = jnp.transpose(inputs[0], (0, 2, 3, 1))
    params = inputs[1:]
    for i, padding in enumerate(CONFIG['paddings']):
        x = _deconv(x, params[2 * i], params[2 * i + 1], padding)
    return jnp.transpose(x, (0, 3, 1, 2))
