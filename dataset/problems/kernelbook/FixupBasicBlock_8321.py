"""FixupBasicBlock: (x+b1a) -> conv3x3 -> (+b1b) relu -> (+b2a) conv3x3 -> *scale+b2b -> +x -> relu."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'FixupBasicBlock_8321',
    'batch': 8,
    'inplanes': 128,
    'planes': 128,
    'height': 128,
    'width': 128,
    'kernel_size': 3,
    'stride': 1,
    'padding': 1,
}


def create_inputs(dtype=jnp.float32):
    k = jax.random.key(0)
    kx, k1, k2 = jax.random.split(k, 3)
    c_in, c_out, ks = CONFIG['inplanes'], CONFIG['planes'], CONFIG['kernel_size']
    x = jax.random.uniform(
        kx, (CONFIG['batch'], c_in, CONFIG['height'], CONFIG['width']), dtype)
    w1 = jax.random.normal(k1, (c_out, c_in, ks, ks), dtype) * 0.02
    w2 = jax.random.normal(k2, (c_out, c_out, ks, ks), dtype) * 0.02
    bias1a = jnp.zeros((1,), dtype)
    bias1b = jnp.zeros((1,), dtype)
    bias2a = jnp.zeros((1,), dtype)
    scale = jnp.ones((1,), dtype)
    bias2b = jnp.zeros((1,), dtype)
    return (x, bias1a, w1, bias1b, bias2a, w2, scale, bias2b)


def _conv3x3(x_nhwc, w_oihw):
    p = CONFIG['padding']
    s = CONFIG['stride']
    return lax.conv_general_dilated(
        x_nhwc, w_oihw,
        window_strides=(s, s),
        padding=((p, p), (p, p)),
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
        precision=lax.Precision.HIGHEST,
    )


def workload(x, bias1a, w1, bias1b, bias2a, w2, scale, bias2b):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    out = _conv3x3(x_nhwc + bias1a, w1)
    out = jax.nn.relu(out + bias1b)
    out = _conv3x3(out + bias2a, w2)
    out = out * scale + bias2b
    out = jax.nn.relu(out + x_nhwc)
    return jnp.transpose(out, (0, 3, 1, 2))
