"""[bf16 variant] ConvNet_16737: four stacked 2x2 valid Conv2d layers (8->16->32->16->8 channels) with bias."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConvNet_16737',
    'batch': 16,
    'in_channels': 8,
    'height': 512,
    'width': 512,
    'channels': (8, 16, 32, 16, 8),
    'kernel_size': 2,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['in_channels'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    k = CONFIG['kernel_size']
    chans = CONFIG['channels']
    params = []
    for i in range(len(chans) - 1):
        cin, cout = chans[i], chans[i + 1]
        w = jax.random.normal(keys[1 + 2 * i], (cout, cin, k, k), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (cout,), dtype) * 0.02
        params.extend([w, b])
    return (x, *params)


def _conv(x_nhwc, w_oihw, b):
    y = lax.conv_general_dilated(
        x_nhwc,
        w_oihw,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )
    return y + b


def workload(x, *params):
    h = jnp.transpose(x, (0, 2, 3, 1))
    for i in range(0, len(params), 2):
        h = _conv(h, params[i], params[i + 1])
    return jnp.transpose(h, (0, 3, 1, 2))
