"""[bf16 variant] DisConvModule: 4x [Conv2d(k5, s2, p2) + bias -> LeakyReLU(0.2)]."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'DisConvModule_13135',
    'batch': 16,
    'input_dim': 4,
    'cnum': 128,
    'height': 512,
    'width': 512,
    'kernel_size': 5,
    'stride': 2,
    'padding': 2,
    'negative_slope': 0.2,
}


def _channels():
    c = CONFIG['cnum']
    return [CONFIG['input_dim'], c, c * 2, c * 4, c * 4]


def create_inputs(dtype=jnp.bfloat16):
    chans = _channels()
    k = CONFIG['kernel_size']
    keys = jax.random.split(jax.random.key(0), 1 + 2 * (len(chans) - 1))
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['input_dim'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    params = []
    for i in range(len(chans) - 1):
        cin, cout = chans[i], chans[i + 1]
        w = jax.random.normal(keys[1 + 2 * i], (k, k, cin, cout), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (cout,), dtype) * 0.02
        params += [w, b]
    return (x, *params)


def workload(x, *params):
    s = CONFIG['stride']
    p = CONFIG['padding']
    slope = CONFIG['negative_slope']
    h = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
    for i in range(0, len(params), 2):
        w, b = params[i], params[i + 1]
        h = lax.conv_general_dilated(
            h, w,
            window_strides=(s, s),
            padding=((p, p), (p, p)),
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        ) + b
        h = jnp.where(h >= 0, h, slope * h)
    return jnp.transpose(h, (0, 3, 1, 2))  # NHWC -> NCHW
