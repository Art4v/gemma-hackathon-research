"""[bf16 variant] Multi-head linear (efficient) attention: QKV linear -> K softmax over seq, Q softmax over features -> KV -> output linear."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'efficient_attention_softmax_kernel',
    'batch': 8,
    'seq_len': 2048,
    'dim_model': 1024,
    'num_heads': 4,
}

_HI = lax.Precision.HIGHEST


def create_inputs(dtype=jnp.bfloat16):
    b, t, d = CONFIG['batch'], CONFIG['seq_len'], CONFIG['dim_model']
    ks = jax.random.split(jax.random.key(0), 11)
    q = jax.random.uniform(ks[0], (b, t, d), dtype)
    k = jax.random.uniform(ks[1], (b, t, d), dtype)
    v = jax.random.uniform(ks[2], (b, t, d), dtype)
    params = []
    for i in range(4):
        params.append(jax.random.normal(ks[3 + 2 * i], (d, d), dtype) * 0.02)
        params.append(jax.random.normal(ks[4 + 2 * i], (d,), dtype) * 0.02)
    return (q, k, v, *params)


def _linear(x, w, b):
    return jnp.matmul(x, w, precision=_HI) + b


def workload(q, k, v, wq, bq, wk, bk, wv, bv, wo, bo):
    batch = q.shape[0]
    d = wq.shape[1]
    h = CONFIG['num_heads']
    dh = d // h

    def heads(x):
        return x.reshape(batch, -1, h, dh).transpose(0, 2, 1, 3)

    qh = heads(_linear(q, wq, bq))
    kh = heads(_linear(k, wk, bk))
    vh = heads(_linear(v, wv, bv))
    scale = dh ** 0.25
    k_sm = jax.nn.softmax(kh / scale, axis=-2)
    kv = jnp.matmul(jnp.swapaxes(k_sm, -1, -2), vh, precision=_HI)
    q_sm = jax.nn.softmax(qh / scale, axis=-1)
    o = jnp.matmul(q_sm, kv, precision=_HI)
    o = o.transpose(0, 2, 1, 3).reshape(batch, -1, d)
    return _linear(o, wo, bo), kv
