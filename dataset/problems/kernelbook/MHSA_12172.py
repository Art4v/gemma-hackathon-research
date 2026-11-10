"""MHSA: 1x1 conv q/k/v -> q^T k + rel_pos^T q -> softmax -> v @ attn^T."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'MHSA_12172',
    'batch': 32,
    'n_dims': 512,
    'heads': 4,
    'width': 32,
    'height': 32,
    'rel_width': 32,
    'rel_height': 32,
}


def create_inputs(dtype=jnp.float32):
    b, c, h = CONFIG['batch'], CONFIG['n_dims'], CONFIG['heads']
    w, hh = CONFIG['width'], CONFIG['height']
    rw, rh = CONFIG['rel_width'], CONFIG['rel_height']
    d = c // h
    ks = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(ks[0], (b, c, w, hh), dtype)
    wq = jax.random.normal(ks[1], (1, 1, c, c), dtype) * 0.02
    bq = jax.random.normal(ks[2], (c,), dtype) * 0.02
    wk = jax.random.normal(ks[3], (1, 1, c, c), dtype) * 0.02
    bk = jax.random.normal(ks[4], (c,), dtype) * 0.02
    wv = jax.random.normal(ks[5], (1, 1, c, c), dtype) * 0.02
    bv = jax.random.normal(ks[6], (c,), dtype) * 0.02
    rel_h = jax.random.normal(ks[7], (1, h, d, 1, rh), dtype)
    rel_w = jax.random.normal(ks[8], (1, h, d, rw, 1), dtype)
    return (x, wq, bq, wk, bk, wv, bv, rel_h, rel_w)


def _conv1x1(x_nhwc, w, b):
    y = lax.conv_general_dilated(
        x_nhwc, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def workload(x, wq, bq, wk, bk, wv, bv, rel_h, rel_w):
    heads = CONFIG['heads']
    bsz, c, w, h = x.shape
    d = c // heads
    n = w * h
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))

    def proj(wt, bs):
        y = jnp.transpose(_conv1x1(x_nhwc, wt, bs), (0, 3, 1, 2))
        return y.reshape(bsz, heads, d, n)

    q, k, v = proj(wq, bq), proj(wk, bk), proj(wv, bv)
    content_content = jnp.einsum('bhdn,bhdm->bhnm', q, k)
    rel = (rel_h + rel_w).reshape(heads, d, -1)[:, :, :n]
    content_position = jnp.einsum('hdm,bhdn->bhmn', rel, q)
    energy = content_content + content_position
    attention = jax.nn.softmax(energy, axis=-1)
    out = jnp.einsum('bhdm,bhnm->bhdn', v, attention)
    return out.reshape(bsz, c, w, h)
