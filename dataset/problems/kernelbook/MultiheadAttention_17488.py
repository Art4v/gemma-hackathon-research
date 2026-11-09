"""[bf16 variant] Multi-head attention over channel-first tensors: reshape heads, bmm(K^T, Q), softmax over keys, bmm(V, W)."""
import math

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MultiheadAttention_17488',
    'batch': 8,
    'd': 512,
    'd_out': 512,
    'n': 2048,
    'm': 2048,
    'num_heads': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    kk, kq, kv = jax.random.split(jax.random.key(0), 3)
    b, d, do = CONFIG['batch'], CONFIG['d'], CONFIG['d_out']
    n, m = CONFIG['n'], CONFIG['m']
    key = jax.random.uniform(kk, (b, d, n), dtype)
    query = jax.random.uniform(kq, (b, d, m), dtype)
    value = jax.random.uniform(kv, (b, do, m), dtype)
    return key, query, value


def workload(key, query, value):
    h = CONFIG['num_heads']
    b, d, n = key.shape
    m = query.shape[2]
    do = value.shape[1]
    k = key.reshape(b * h, d // h, n)
    q = query.reshape(b * h, d // h, m)
    v = value.reshape(b * h, do // h, m)
    affinity = jnp.einsum('bcn,bcm->bnm', k, q, precision=jax.lax.Precision.HIGHEST)
    weight = jax.nn.softmax(affinity / math.sqrt(d), axis=1)
    # torch.bmm(value, weight): contracts value's last dim (m) with weight's dim 1 (n); requires n == m.
    out = jnp.einsum('bcj,bjk->bck', v, weight, precision=jax.lax.Precision.HIGHEST)
    return out.reshape(b, -1, m), weight.reshape(b, h, n, m)
