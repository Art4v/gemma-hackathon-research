"""[bf16 variant] ConvDenoiser: 3x(conv3x3+ReLU+maxpool2) -> 3x(conv_transpose+ReLU) -> conv3x3 -> sigmoid."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ConvDenoiser_3791',
    'batch': 32,
    'in_channels': 3,
    'height': 512,
    'width': 512,
    'c1': 32,
    'c2': 16,
    'c3': 8,
    't1': 8,
    't2': 16,
    't3': 32,
    'out_channels': 3,
    'conv_kernel': 3,
    'conv_padding': 1,
    'pool': 2,
    't1_kernel': 2,
    't2_kernel': 2,
    't3_kernel': 3,
    't_stride': 2,
}

_DN = ('NHWC', 'HWIO', 'NHWC')


def _conv(x, w, b):
    # w is PyTorch OIHW
    p = CONFIG['conv_padding']
    w_hwio = jnp.transpose(w, (2, 3, 1, 0))
    y = lax.conv_general_dilated(
        x, w_hwio, (1, 1), ((p, p), (p, p)),
        dimension_numbers=_DN, precision=lax.Precision.HIGHEST)
    return y + b


def _tconv(x, w, b):
    # w is PyTorch ConvTranspose IOHW; transposed conv == dilated input, flipped kernel, pad k-1
    s = CONFIG['t_stride']
    k = w.shape[2]
    w_hwio = jnp.transpose(w[:, :, ::-1, ::-1], (2, 3, 0, 1))
    y = lax.conv_general_dilated(
        x, w_hwio, (1, 1), ((k - 1, k - 1), (k - 1, k - 1)),
        lhs_dilation=(s, s), dimension_numbers=_DN,
        precision=lax.Precision.HIGHEST)
    return y + b


def _pool(x):
    p = CONFIG['pool']
    return lax.reduce_window(x, -jnp.inf, lax.max, (1, p, p, 1), (1, p, p, 1), 'VALID')


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    keys = jax.random.split(jax.random.key(0), 15)
    x = jax.random.uniform(keys[0], (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    k = c['conv_kernel']
    shapes = [
        (c['c1'], c['in_channels'], k, k),             # conv1 OIHW
        (c['c2'], c['c1'], k, k),                      # conv2
        (c['c3'], c['c2'], k, k),                      # conv3
        (c['c3'], c['t1'], c['t1_kernel'], c['t1_kernel']),  # t_conv1 IOHW
        (c['t1'], c['t2'], c['t2_kernel'], c['t2_kernel']),  # t_conv2
        (c['t2'], c['t3'], c['t3_kernel'], c['t3_kernel']),  # t_conv3
        (c['out_channels'], c['t3'], k, k),            # conv_out OIHW
    ]
    bias_sizes = [c['c1'], c['c2'], c['c3'], c['t1'], c['t2'], c['t3'], c['out_channels']]
    params = []
    for i, (ws, bs) in enumerate(zip(shapes, bias_sizes)):
        params.append(jax.random.normal(keys[1 + 2 * i], ws, dtype) * 0.02)
        params.append(jax.random.normal(keys[2 + 2 * i], (bs,), dtype) * 0.02)
    return (x, *params)


def workload(x, w1, b1, w2, b2, w3, b3, tw1, tb1, tw2, tb2, tw3, tb3, wo, bo):
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = _pool(jax.nn.relu(_conv(h, w1, b1)))
    h = _pool(jax.nn.relu(_conv(h, w2, b2)))
    h = _pool(jax.nn.relu(_conv(h, w3, b3)))
    h = jax.nn.relu(_tconv(h, tw1, tb1))
    h = jax.nn.relu(_tconv(h, tw2, tb2))
    h = jax.nn.relu(_tconv(h, tw3, tb3))
    h = jax.nn.sigmoid(_conv(h, wo, bo))
    return jnp.transpose(h, (0, 3, 1, 2))
