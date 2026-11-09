"""Conv2dNormActiv_7936: Conv2d(3x3, pad 1) -> GroupNorm(out_ch/4 groups) -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Conv2dNormActiv_7936',
    'batch': 16,
    'in_ch': 128,
    'out_ch': 128,
    'height': 128,
    'width': 128,
    'k_size': 3,
    'stride': 1,
    'dilation': 1,
    'channels_per_group': 4,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k_x, k_w, k_b = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(k_x, (c['batch'], c['in_ch'], c['height'], c['width']), dtype)
    w = jax.random.normal(k_w, (c['out_ch'], c['in_ch'], c['k_size'], c['k_size']), dtype) * 0.02
    b = jax.random.normal(k_b, (c['out_ch'],), dtype) * 0.02
    gamma = jnp.ones((c['out_ch'],), dtype)
    beta = jnp.zeros((c['out_ch'],), dtype)
    return x, w, b, gamma, beta


def workload(x, w, b, gamma, beta):
    c = CONFIG
    k = w.shape[2]
    pad = k // 2 if k >= 2 else 0
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    w_hwio = jnp.transpose(w, (2, 3, 1, 0))
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio,
        window_strides=(c['stride'], c['stride']),
        padding=((pad, pad), (pad, pad)),
        rhs_dilation=(c['dilation'], c['dilation']),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    ) + b
    n, h, wd, ch = y.shape
    groups = ch // c['channels_per_group']
    yg = y.reshape(n, h, wd, groups, ch // groups)
    mean = jnp.mean(yg, axis=(1, 2, 4), keepdims=True)
    var = jnp.mean(jnp.square(yg - mean), axis=(1, 2, 4), keepdims=True)
    yg = (yg - mean) * lax.rsqrt(var + c['eps'])
    y = yg.reshape(n, h, wd, ch) * gamma + beta
    y = jnp.maximum(y, 0)
    return jnp.transpose(y, (0, 3, 1, 2))
