"""Conv2d -> GroupNorm -> ReLU -> Conv2d -> GroupNorm -> ReLU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Conv2d_GN_ReLUx2_2246',
    'batch': 16,
    'in_channels': 128,
    'out_channels': 128,
    'height': 128,
    'width': 128,
    'num_groups': 1,
    'ksize': 3,
    'stride': 1,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    k_x, k_w1, k_w2 = jax.random.split(jax.random.key(0), 3)
    x = jax.random.uniform(k_x, (c['batch'], c['in_channels'], c['height'], c['width']), dtype)
    w1 = jax.random.normal(k_w1, (c['ksize'], c['ksize'], c['in_channels'], c['out_channels']), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (c['ksize'], c['ksize'], c['out_channels'], c['out_channels']), dtype) * 0.02
    g1 = jnp.ones((c['out_channels'],), dtype)
    b1 = jnp.zeros((c['out_channels'],), dtype)
    g2 = jnp.ones((c['out_channels'],), dtype)
    b2 = jnp.zeros((c['out_channels'],), dtype)
    return x, w1, g1, b1, w2, g2, b2


def _conv_gn_relu(x, w, gamma, beta):
    # x: NHWC, w: HWIO
    k = w.shape[0]
    pad = 0 if k < 2 else k // 2
    s = CONFIG['stride']
    y = lax.conv_general_dilated(
        x, w, window_strides=(s, s), padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    n, h, wd, c = y.shape
    g = CONFIG['num_groups']
    yg = y.reshape(n, h, wd, g, c // g)
    mean = jnp.mean(yg, axis=(1, 2, 4), keepdims=True)
    var = jnp.mean(jnp.square(yg - mean), axis=(1, 2, 4), keepdims=True)
    yn = ((yg - mean) * lax.rsqrt(var + CONFIG['eps'])).reshape(n, h, wd, c)
    return jax.nn.relu(yn * gamma + beta)


def workload(x, w1, g1, b1, w2, g2, b2):
    out = jnp.transpose(x, (0, 2, 3, 1))
    out = _conv_gn_relu(out, w1, g1, b1)
    out = _conv_gn_relu(out, w2, g2, b2)
    return jnp.transpose(out, (0, 3, 1, 2))
