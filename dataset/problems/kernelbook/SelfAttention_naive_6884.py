"""[bf16 variant] Single-head self-attention: QKV linear projections -> QK^T / sqrt(dim_emb) -> softmax -> @V."""
import math

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SelfAttention_naive_6884',
    'batch': 16,
    'seq_len': 2048,
    'dim_emb': 1024,
    'dim_internal': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 7)
    b, t = CONFIG['batch'], CONFIG['seq_len']
    e, d = CONFIG['dim_emb'], CONFIG['dim_internal']
    x = jax.random.uniform(keys[0], (b, t, e), dtype)
    wq = jax.random.normal(keys[1], (e, d), dtype) * 0.02
    bq = jax.random.normal(keys[2], (d,), dtype) * 0.02
    wk = jax.random.normal(keys[3], (e, d), dtype) * 0.02
    bk = jax.random.normal(keys[4], (d,), dtype) * 0.02
    wv = jax.random.normal(keys[5], (e, d), dtype) * 0.02
    bv = jax.random.normal(keys[6], (d,), dtype) * 0.02
    return x, wq, bq, wk, bk, wv, bv


def workload(x, wq, bq, wk, bk, wv, bv):
    q = x @ wq + bq
    k = x @ wk + bk
    v = x @ wv + bv
    # The scale uses dim_emb, matching the PyTorch module's kSqrt_dim_emb.
    scores = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) / math.sqrt(x.shape[-1])
    p = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(p, v)
