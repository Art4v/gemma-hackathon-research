"""[bf16 variant] SANet: mean-variance norm -> 1x1 convs f/g/h -> softmax(F^T G) attention -> H S^T -> 1x1 out_conv + content residual."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SANet_8218',
    'batch': 8,
    'channels': 1024,
    'height': 64,
    'width': 64,
    'kernel_size': (1, 1),
    'strides': (1, 1),
    'padding': ((0, 0), (0, 0)),
    'eps': 1e-05,
    'ddof': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    b, c = CONFIG['batch'], CONFIG['channels']
    h, w = CONFIG['height'], CONFIG['width']
    kh, kw = CONFIG['kernel_size']
    scale = CONFIG['weight_scale']
    keys = jax.random.split(jax.random.key(0), 10)
    content = jax.random.uniform(keys[0], (b, c, h, w), dtype)
    style = jax.random.uniform(keys[1], (b, c, h, w), dtype)
    f_w = jax.random.normal(keys[2], (c, c, kh, kw), dtype) * scale
    f_b = jax.random.normal(keys[3], (c,), dtype) * scale
    g_w = jax.random.normal(keys[4], (c, c, kh, kw), dtype) * scale
    g_b = jax.random.normal(keys[5], (c,), dtype) * scale
    h_w = jax.random.normal(keys[6], (c, c, kh, kw), dtype) * scale
    h_b = jax.random.normal(keys[7], (c,), dtype) * scale
    out_w = jax.random.normal(keys[8], (c, c, kh, kw), dtype) * scale
    out_b = jax.random.normal(keys[9], (c,), dtype) * scale
    return (content, style, f_w, f_b, g_w, g_b, h_w, h_b, out_w, out_b)


def _mean_variance_norm(feat):
    b, c = feat.shape[:2]
    flat = feat.reshape(b, c, -1)
    var = jnp.var(flat, axis=2, ddof=CONFIG['ddof']) + CONFIG['eps']
    std = jnp.sqrt(var).reshape(b, c, 1, 1)
    mean = jnp.mean(flat, axis=2).reshape(b, c, 1, 1)
    return (feat - mean) / std


def _conv(x, weight, bias):
    x = jnp.transpose(x, (0, 2, 3, 1))
    y = lax.conv_general_dilated(
        x,
        weight,
        window_strides=CONFIG['strides'],
        padding=CONFIG['padding'],
        dimension_numbers=('NHWC', 'OIHW', 'NHWC'),
    )
    y = y + bias
    return jnp.transpose(y, (0, 3, 1, 2))


def workload(content, style, f_w, f_b, g_w, g_b, h_w, h_b, out_w, out_b):
    b, c, h, w = content.shape
    F = _conv(_mean_variance_norm(content), f_w, f_b)
    G = _conv(_mean_variance_norm(style), g_w, g_b)
    H = _conv(style, h_w, h_b)
    F = jnp.transpose(F.reshape(b, -1, h * w), (0, 2, 1))
    G = G.reshape(b, -1, h * w)
    S = jax.nn.softmax(jnp.matmul(F, G), axis=-1)
    H = H.reshape(b, -1, h * w)
    O = jnp.matmul(H, jnp.transpose(S, (0, 2, 1)))
    O = O.reshape(b, c, h, w)
    return _conv(O, out_w, out_b) + content
