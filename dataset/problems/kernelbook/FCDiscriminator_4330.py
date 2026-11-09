"""[bf16 variant] FCDiscriminator: 4x (conv3x3 s2 p1) with ReLU/ReLU/ReLU/LeakyReLU(0.2), then conv1x1 classifier."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'FCDiscriminator_4330',
    'batch': 16,
    'inplanes': 4,
    'planes': 64,
    'height': 512,
    'width': 512,
    'kernel_size': 3,
    'stride': 2,
    'padding': 1,
    'negative_slope': 0.2,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 11)
    c, p, k = CONFIG['inplanes'], CONFIG['planes'], CONFIG['kernel_size']
    x = jax.random.uniform(keys[0], (CONFIG['batch'], c, CONFIG['height'], CONFIG['width']), dtype)
    chans = [c, p, p * 2, p * 4, p * 8]
    params = []
    for i in range(4):
        params.append(jax.random.normal(keys[1 + 2 * i], (k, k, chans[i], chans[i + 1]), dtype) * 0.02)
        params.append(jax.random.normal(keys[2 + 2 * i], (chans[i + 1],), dtype) * 0.02)
    params.append(jax.random.normal(keys[9], (1, 1, chans[4], 1), dtype) * 0.02)
    params.append(jax.random.normal(keys[10], (1,), dtype) * 0.02)
    return (x, *params)


def _conv(x, w, b, stride, pad):
    y = lax.conv_general_dilated(
        x, w, window_strides=(stride, stride), padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'), precision=lax.Precision.HIGHEST)
    return y + b


def workload(x, w1, b1, w2, b2, w3, b3, w4, b4, wc, bc):
    s, p = CONFIG['stride'], CONFIG['padding']
    h = jnp.transpose(x, (0, 2, 3, 1))
    h = jax.nn.relu(_conv(h, w1, b1, s, p))
    h = jax.nn.relu(_conv(h, w2, b2, s, p))
    h = jax.nn.relu(_conv(h, w3, b3, s, p))
    h = jax.nn.leaky_relu(_conv(h, w4, b4, s, p), negative_slope=CONFIG['negative_slope'])
    h = _conv(h, wc, bc, 1, 0)
    return jnp.transpose(h, (0, 3, 1, 2))
