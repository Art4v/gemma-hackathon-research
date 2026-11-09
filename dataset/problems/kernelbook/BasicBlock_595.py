"""BasicBlock: conv3x3 -> ReLU -> conv3x3 -> + identity residual -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'BasicBlock_595',
    'batch': 8,
    'inplanes': 128,
    'planes': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 1,
    'dilation': 1,
}


def create_inputs(dtype=jnp.float32):
    k = jax.random.key(0)
    kx, k1, k2 = jax.random.split(k, 3)
    c_in, c_out, ks = CONFIG['inplanes'], CONFIG['planes'], CONFIG['kernel_size']
    x = jax.random.uniform(
        kx, (CONFIG['batch'], c_in, CONFIG['height'], CONFIG['width']), dtype)
    w1 = jax.random.normal(k1, (ks, ks, c_in, c_out), dtype) * 0.02
    w2 = jax.random.normal(k2, (ks, ks, c_out, c_out), dtype) * 0.02
    return x, w1, w2


def _conv3x3(x_nhwc, w_hwio, stride):
    d = CONFIG['dilation']
    return lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(stride, stride),
        padding=((d, d), (d, d)),
        rhs_dilation=(d, d),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )


def workload(x, w1, w2):
    h = jnp.transpose(x, (0, 2, 3, 1))
    out = jax.nn.relu(_conv3x3(h, w1, CONFIG['stride']))
    out = _conv3x3(out, w2, 1)
    out = jax.nn.relu(out + h)
    return jnp.transpose(out, (0, 3, 1, 2))
