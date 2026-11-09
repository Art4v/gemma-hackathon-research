"""[bf16 variant] ParallelDilatedConv: 4 parallel dilated 3x3 Conv2d (dilation 1..4, same padding) -> ELU each -> sum."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ParallelDilatedConv_3441',
    'batch': 8,
    'inplanes': 128,
    'planes': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'dilations': (1, 2, 3, 4),
    'elu_alpha': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    n_branch = len(c['dilations'])
    keys = jax.random.split(jax.random.key(0), 1 + 2 * n_branch)
    x = jax.random.uniform(keys[0], (c['batch'], c['inplanes'], c['height'], c['width']), dtype)
    k = c['kernel_size']
    params = []
    for i in range(n_branch):
        w = jax.random.normal(keys[1 + 2 * i], (k, k, c['inplanes'], c['planes']), dtype) * 0.02
        b = jax.random.normal(keys[2 + 2 * i], (c['planes'],), dtype) * 0.02
        params += [w, b]
    return (x, *params)


def workload(x, *params):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    out = None
    for i, d in enumerate(CONFIG['dilations']):
        w, b = params[2 * i], params[2 * i + 1]
        y = lax.conv_general_dilated(
            x_nhwc, w,
            window_strides=(1, 1),
            padding=((d, d), (d, d)),
            rhs_dilation=(d, d),
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        ) + b
        y = jax.nn.elu(y, alpha=CONFIG['elu_alpha'])
        out = y if out is None else out + y
    return jnp.transpose(out, (0, 3, 1, 2))
