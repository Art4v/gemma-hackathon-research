"""[bf16 variant] PAM_Module_15973: 1x1 conv Q/K/V -> QK^T softmax attention -> AV -> 1x1 conv W -> + x."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'PAM_Module_15973',
    'batch': 8,
    'in_dim': 512,
    'reduction': 4,
    'height': 64,
    'width': 64,
}


def create_inputs(dtype=jnp.bfloat16):
    b, c, h, w = CONFIG['batch'], CONFIG['in_dim'], CONFIG['height'], CONFIG['width']
    cr = c // CONFIG['reduction']
    keys = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(keys[0], (b, c, h, w), dtype)
    wq = jax.random.normal(keys[1], (1, 1, c, cr), dtype) * 0.02
    bq = jax.random.normal(keys[2], (cr,), dtype) * 0.02
    wk = jax.random.normal(keys[3], (1, 1, c, cr), dtype) * 0.02
    bk = jax.random.normal(keys[4], (cr,), dtype) * 0.02
    wv = jax.random.normal(keys[5], (1, 1, c, cr), dtype) * 0.02
    bv = jax.random.normal(keys[6], (cr,), dtype) * 0.02
    ww = jax.random.normal(keys[7], (1, 1, cr, c), dtype) * 0.02
    bw = jax.random.normal(keys[8], (c,), dtype) * 0.02
    return x, wq, bq, wk, bk, wv, bv, ww, bw


def _conv1x1(x_nhwc, w_hwio, b):
    y = lax.conv_general_dilated(
        x_nhwc, w_hwio, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def workload(x, wq, bq, wk, bk, wv, bv, ww, bw):
    b, c, h, w = x.shape
    n = h * w
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    q = _conv1x1(x_nhwc, wq, bq).reshape(b, n, -1)
    k = _conv1x1(x_nhwc, wk, bk).reshape(b, n, -1)
    v = _conv1x1(x_nhwc, wv, bv).reshape(b, n, -1)
    energy = jnp.einsum('bic,bjc->bij', q, k)
    attention = jax.nn.softmax(energy, axis=-1)
    out = jnp.einsum('bij,bjc->bic', attention, v).reshape(b, h, w, -1)
    out = _conv1x1(out, ww, bw)
    return jnp.transpose(out, (0, 3, 1, 2)) + x
