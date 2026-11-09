"""[bf16 variant] SANet: mean_variance_norm -> 1x1 convs f/g/h -> spatial attention (bmm, softmax, bmm) -> out_conv -> residual."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'SANet_5899',
    'batch': 8,
    'in_planes': 512,
    'height': 64,
    'width': 64,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    b, c, h, w = CONFIG['batch'], CONFIG['in_planes'], CONFIG['height'], CONFIG['width']
    keys = jax.random.split(jax.random.key(0), 10)
    content = jax.random.uniform(keys[0], (b, c, h, w), dtype)
    style = jax.random.uniform(keys[1], (b, c, h, w), dtype)
    params = []
    for i in range(4):
        params.append(jax.random.normal(keys[2 + 2 * i], (1, 1, c, c), dtype) * 0.02)
        params.append(jax.random.normal(keys[3 + 2 * i], (c,), dtype) * 0.02)
    return (content, style, *params)


def _mean_variance_norm(x_nhwc):
    # torch .var defaults to unbiased (ddof=1)
    mean = jnp.mean(x_nhwc, axis=(1, 2), keepdims=True)
    var = jnp.var(x_nhwc, axis=(1, 2), keepdims=True, ddof=1) + CONFIG['eps']
    return (x_nhwc - mean) / jnp.sqrt(var)


def _conv1x1(x_nhwc, w_hwio, bias):
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + bias


def workload(content, style, wf, bf, wg, bg, wh, bh, wo, bo):
    b, c, h, w = content.shape
    _, _, hs, ws = style.shape
    content_n = jnp.transpose(content, (0, 2, 3, 1))
    style_n = jnp.transpose(style, (0, 2, 3, 1))

    f = _conv1x1(_mean_variance_norm(content_n), wf, bf).reshape(b, h * w, c)
    g = _conv1x1(_mean_variance_norm(style_n), wg, bg).reshape(b, hs * ws, c)
    hv = _conv1x1(style_n, wh, bh).reshape(b, hs * ws, c)

    s = jax.nn.softmax(jnp.einsum('bpc,bqc->bpq', f, g), axis=-1)
    o = jnp.einsum('bpq,bqc->bpc', s, hv).reshape(b, h, w, c)
    o = _conv1x1(o, wo, bo) + content_n
    return jnp.transpose(o, (0, 3, 1, 2))
