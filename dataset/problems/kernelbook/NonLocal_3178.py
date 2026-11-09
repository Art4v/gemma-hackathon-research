"""[bf16 variant] NonLocal block: 1x1 conv Q/K/V -> softmax(K^T Q over keys) -> V @ A -> 1x1 conv project + residual."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'NonLocal_3178',
    'batch': 8,
    'in_size': 1024,
    'attention_size': 128,
    'height': 64,
    'width': 64,
}


def create_inputs(dtype=jnp.bfloat16):
    b, c, a = CONFIG['batch'], CONFIG['in_size'], CONFIG['attention_size']
    h, w = CONFIG['height'], CONFIG['width']
    keys = jax.random.split(jax.random.key(0), 9)
    x = jax.random.uniform(keys[0], (b, c, h, w), dtype)
    wq = jax.random.normal(keys[1], (1, 1, c, a), dtype) * 0.02
    bq = jax.random.normal(keys[2], (a,), dtype) * 0.02
    wk = jax.random.normal(keys[3], (1, 1, c, a), dtype) * 0.02
    bk = jax.random.normal(keys[4], (a,), dtype) * 0.02
    wv = jax.random.normal(keys[5], (1, 1, c, a), dtype) * 0.02
    bv = jax.random.normal(keys[6], (a,), dtype) * 0.02
    wp = jax.random.normal(keys[7], (1, 1, a, c), dtype) * 0.02
    bp = jax.random.normal(keys[8], (c,), dtype) * 0.02
    return x, wq, bq, wk, bk, wv, bv, wp, bp


def _conv1x1(x_nhwc, w, b):
    y = lax.conv_general_dilated(
        x_nhwc, w, window_strides=(1, 1), padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
    return y + b


def workload(x, wq, bq, wk, bk, wv, bv, wp, bp):
    bsz, _, h, w = x.shape
    a = wq.shape[-1]
    n = h * w
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    q = _conv1x1(x_nhwc, wq, bq).reshape(bsz, n, a)
    k = _conv1x1(x_nhwc, wk, bk).reshape(bsz, n, a)
    v = _conv1x1(x_nhwc, wv, bv).reshape(bsz, n, a)
    # logits[b, key, query]; torch softmaxes over dim=1, the key axis.
    logits = jnp.einsum('bka,bqa->bkq', k, q)
    assign = jax.nn.softmax(logits, axis=1)
    result = jnp.einsum('bka,bkq->bqa', v, assign).reshape(bsz, h, w, a)
    out = _conv1x1(result, wp, bp) + x_nhwc
    return jnp.transpose(out, (0, 3, 1, 2))
