"""[bf16 variant] GlobalConvBlock: (conv kx1 -> conv 1xk) + (conv 1xk -> conv kx1), NCHW interface."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'GlobalConvBlock_9764',
    'batch': 8,
    'in_dim': 128,
    'out_dim': 128,
    'height': 128,
    'width': 128,
    'kernel_size': (4, 4),
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    k0, k1 = c['kernel_size']
    cin, cout = c['in_dim'], c['out_dim']
    keys = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(keys[0], (c['batch'], cin, c['height'], c['width']), dtype)
    # HWIO weights
    w_l1 = jax.random.normal(keys[1], (k0, 1, cin, cout), dtype) * 0.02
    b_l1 = jax.random.normal(keys[2], (cout,), dtype) * 0.02
    w_l2 = jax.random.normal(keys[3], (1, k1, cout, cout), dtype) * 0.02
    b_l2 = jax.random.normal(keys[4], (cout,), dtype) * 0.02
    w_r1 = jax.random.normal(keys[5], (1, k1, cin, cout), dtype) * 0.02
    b_r1 = jax.random.normal(keys[6], (cout,), dtype) * 0.02
    w_r2 = jax.random.normal(keys[7], (k0, 1, cout, cout), dtype) * 0.02
    b_r2 = jax.random.normal(keys[8], (cout,), dtype) * 0.02
    return x, w_l1, b_l1, w_l2, b_l2, w_r1, b_r1, w_r2, b_r2


def _conv(x, w, b, pad_h, pad_w):
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1),
        padding=((pad_h, pad_h), (pad_w, pad_w)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    return y + b


def workload(x, w_l1, b_l1, w_l2, b_l2, w_r1, b_r1, w_r2, b_r2):
    k0, k1 = CONFIG['kernel_size']
    pad0 = (k0 - 1) // 2
    pad1 = (k1 - 1) // 2
    h = jnp.transpose(x, (0, 2, 3, 1))
    x_l = _conv(_conv(h, w_l1, b_l1, pad0, 0), w_l2, b_l2, 0, pad1)
    x_r = _conv(_conv(h, w_r1, b_r1, 0, pad1), w_r2, b_r2, pad0, 0)
    return jnp.transpose(x_l + x_r, (0, 3, 1, 2))
