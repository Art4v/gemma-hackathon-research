"""ResnetBlock_2170: leaky_relu -> conv3x3+bias -> leaky_relu -> conv3x3+bias -> identity-shortcut residual add (x + 0.1 * dx)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ResnetBlock_2170',
    'batch': 8,
    'fin': 128,
    'fout': 128,
    'fhidden': 128,
    'height': 256,
    'width': 256,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
    'negative_slope': 0.2,
    'residual_scale': 0.1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w0, k_b0, k_w1, k_b1 = jax.random.split(key, 5)
    ks = CONFIG['kernel_size']
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(
        k_x,
        (CONFIG['batch'], CONFIG['fin'], CONFIG['height'], CONFIG['width']),
        dtype,
    )
    w0 = jax.random.normal(k_w0, (CONFIG['fhidden'], CONFIG['fin'], ks, ks), dtype) * scale
    b0 = jax.random.normal(k_b0, (CONFIG['fhidden'],), dtype) * scale
    w1 = jax.random.normal(k_w1, (CONFIG['fout'], CONFIG['fhidden'], ks, ks), dtype) * scale
    b1 = jax.random.normal(k_b1, (CONFIG['fout'],), dtype) * scale
    return x, w0, b0, w1, b1


def _leaky_relu(x):
    return jnp.where(x >= 0, x, CONFIG['negative_slope'] * x)


def _conv(x, w, b):
    stride = CONFIG['stride']
    pad = CONFIG['padding']
    y = lax.conv_general_dilated(
        x,
        w,
        window_strides=(stride, stride),
        padding=((pad, pad), (pad, pad)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    return y + b


def workload(x, w0, b0, w1, b1):
    h = jnp.transpose(x, (0, 2, 3, 1))
    dx = _conv(_leaky_relu(h), w0, b0)
    dx = _conv(_leaky_relu(dx), w1, b1)
    out = h + CONFIG['residual_scale'] * dx
    return jnp.transpose(out, (0, 3, 1, 2))
