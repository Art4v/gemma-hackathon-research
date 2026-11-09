"""[bf16 variant] Fire: squeeze conv1x1 + ReLU -> concat(expand conv1x1 + ReLU, expand conv3x3 + ReLU)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'Fire_9590',
    'batch': 16,
    'inplanes': 512,
    'squeeze_planes': 128,
    'expand1x1_planes': 256,
    'expand3x3_planes': 256,
    'height': 64,
    'width': 64,
    'expand3x3_padding': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 7)
    x = jax.random.uniform(ks[0], (c['batch'], c['inplanes'], c['height'], c['width']), dtype)
    w_sq = jax.random.normal(ks[1], (1, 1, c['inplanes'], c['squeeze_planes']), dtype) * 0.02
    b_sq = jax.random.normal(ks[2], (c['squeeze_planes'],), dtype) * 0.02
    w_e1 = jax.random.normal(ks[3], (1, 1, c['squeeze_planes'], c['expand1x1_planes']), dtype) * 0.02
    b_e1 = jax.random.normal(ks[4], (c['expand1x1_planes'],), dtype) * 0.02
    w_e3 = jax.random.normal(ks[5], (3, 3, c['squeeze_planes'], c['expand3x3_planes']), dtype) * 0.02
    b_e3 = jax.random.normal(ks[6], (c['expand3x3_planes'],), dtype) * 0.02
    return x, w_sq, b_sq, w_e1, b_e1, w_e3, b_e3


def _conv_relu(x, w, b, pad):
    y = lax.conv_general_dilated(
        x, w, window_strides=(1, 1), padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        precision=lax.Precision.HIGHEST)
    return jax.nn.relu(y + b)


def workload(x, w_sq, b_sq, w_e1, b_e1, w_e3, b_e3):
    p = CONFIG['expand3x3_padding']
    h = jnp.transpose(x, (0, 2, 3, 1))
    s = _conv_relu(h, w_sq, b_sq, 0)
    out = jnp.concatenate([_conv_relu(s, w_e1, b_e1, 0), _conv_relu(s, w_e3, b_e3, p)], axis=-1)
    return jnp.transpose(out, (0, 3, 1, 2))
