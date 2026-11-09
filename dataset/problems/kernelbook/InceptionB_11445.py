"""InceptionB: [conv3x3/s2+relu, conv1x1+relu -> conv3x3/p1+relu -> conv3x3/s2+relu, maxpool3x3/s2] -> concat."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'InceptionB_11445',
    'batch': 16,
    'in_channels': 128,
    'height': 128,
    'width': 128,
    'branch3x3_out': 384,
    'dbl_1_out': 64,
    'dbl_2_out': 96,
    'dbl_3_out': 96,
    'kernel': 3,
    'stride': 2,
    'dbl_2_padding': 1,
    'pool_kernel': 3,
    'pool_stride': 2,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k = c['kernel']
    keys = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(keys[1], (k, k, c['in_channels'], c['branch3x3_out']), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (c['branch3x3_out'],), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (1, 1, c['in_channels'], c['dbl_1_out']), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (c['dbl_1_out'],), dtype) * 0.02
    w3 = jax.random.normal(keys[5], (k, k, c['dbl_1_out'], c['dbl_2_out']), dtype) * 0.02
    b3 = jax.random.normal(keys[6], (c['dbl_2_out'],), dtype) * 0.02
    w4 = jax.random.normal(keys[7], (k, k, c['dbl_2_out'], c['dbl_3_out']), dtype) * 0.02
    b4 = jax.random.normal(keys[8], (c['dbl_3_out'],), dtype) * 0.02
    return x, w1, b1, w2, b2, w3, b3, w4, b4


def _conv_relu(x, w, b, stride, padding):
    y = lax.conv_general_dilated(
        x, w, window_strides=(stride, stride), padding=padding,
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return jax.nn.relu(y + b)


def workload(x, w1, b1, w2, b2, w3, b3, w4, b4):
    s = CONFIG['stride']
    p = CONFIG['dbl_2_padding']
    pk = CONFIG['pool_kernel']
    ps = CONFIG['pool_stride']
    h = jnp.transpose(x, (0, 2, 3, 1))
    branch3x3 = _conv_relu(h, w1, b1, s, 'VALID')
    dbl = _conv_relu(h, w2, b2, 1, 'VALID')
    dbl = _conv_relu(dbl, w3, b3, 1, ((p, p), (p, p)))
    dbl = _conv_relu(dbl, w4, b4, s, 'VALID')
    pool = lax.reduce_window(h, -jnp.inf, lax.max, (1, pk, pk, 1), (1, ps, ps, 1), 'VALID')
    out = jnp.concatenate([branch3x3, dbl, pool.astype(h.dtype)], axis=-1)
    return jnp.transpose(out, (0, 3, 1, 2))
