"""[bf16 variant] TransformerLayer: q/k/v Linear -> MultiheadAttention (+residual) -> fc1 -> fc2 (+residual)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'TransformerLayer_11283',
    'seq_len': 4096,
    'c': 2048,
    'num_heads': 4,
}

_HI = lax.Precision.HIGHEST


def create_inputs(dtype=jnp.bfloat16):
    L, c = CONFIG['seq_len'], CONFIG['c']
    keys = jax.random.split(jax.random.key(0), 8)
    x = jax.random.uniform(keys[0], (L, c), dtype)
    wq = jax.random.normal(keys[1], (c, c), dtype) * 0.02
    wk = jax.random.normal(keys[2], (c, c), dtype) * 0.02
    wv = jax.random.normal(keys[3], (c, c), dtype) * 0.02
    in_w = jax.random.normal(keys[4], (c, 3 * c), dtype) * 0.02
    in_b = jnp.zeros((3 * c,), dtype)
    out_w = jax.random.normal(keys[5], (c, c), dtype) * 0.02
    out_b = jnp.zeros((c,), dtype)
    w1 = jax.random.normal(keys[6], (c, c), dtype) * 0.02
    w2 = jax.random.normal(keys[7], (c, c), dtype) * 0.02
    return x, wq, wk, wv, in_w, in_b, out_w, out_b, w1, w2


def workload(x, wq, wk, wv, in_w, in_b, out_w, out_b, w1, w2):
    L, c = x.shape
    h = CONFIG['num_heads']
    d = c // h
    q = jnp.dot(jnp.dot(x, wq, precision=_HI), in_w[:, :c], precision=_HI) + in_b[:c]
    k = jnp.dot(jnp.dot(x, wk, precision=_HI), in_w[:, c:2 * c], precision=_HI) + in_b[c:2 * c]
    v = jnp.dot(jnp.dot(x, wv, precision=_HI), in_w[:, 2 * c:], precision=_HI) + in_b[2 * c:]
    q = q.reshape(L, h, d) * (1.0 / jnp.sqrt(jnp.asarray(d, x.dtype)))
    k = k.reshape(L, h, d)
    v = v.reshape(L, h, d)
    scores = jnp.einsum('lhd,mhd->hlm', q, k, precision=_HI)
    probs = jax.nn.softmax(scores, axis=-1)
    attn = jnp.einsum('hlm,mhd->lhd', probs, v, precision=_HI).reshape(L, c)
    x = jnp.dot(attn, out_w, precision=_HI) + out_b + x
    x = jnp.dot(jnp.dot(x, w1, precision=_HI), w2, precision=_HI) + x
    return x
