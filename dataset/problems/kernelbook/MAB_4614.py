"""[bf16 variant] MAB: Linear Q/K/V projections -> multi-head softmax attention with Q residual -> O + relu(Linear(O))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MAB_4614',
    'batch': 8,
    'seq_q': 2048,
    'seq_k': 2048,
    'dim_Q': 1024,
    'dim_K': 1024,
    'dim_V': 1024,
    'num_heads': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 10)
    Q = jax.random.uniform(ks[0], (c['batch'], c['seq_q'], c['dim_Q']), dtype)
    K = jax.random.uniform(ks[1], (c['batch'], c['seq_k'], c['dim_K']), dtype)
    wq = jax.random.normal(ks[2], (c['dim_Q'], c['dim_V']), dtype) * 0.02
    bq = jax.random.normal(ks[3], (c['dim_V'],), dtype) * 0.02
    wk = jax.random.normal(ks[4], (c['dim_K'], c['dim_V']), dtype) * 0.02
    bk = jax.random.normal(ks[5], (c['dim_V'],), dtype) * 0.02
    wv = jax.random.normal(ks[6], (c['dim_K'], c['dim_V']), dtype) * 0.02
    bv = jax.random.normal(ks[7], (c['dim_V'],), dtype) * 0.02
    wo = jax.random.normal(ks[8], (c['dim_V'], c['dim_V']), dtype) * 0.02
    bo = jax.random.normal(ks[9], (c['dim_V'],), dtype) * 0.02
    return Q, K, wq, bq, wk, bk, wv, bv, wo, bo


def workload(Q, K, wq, bq, wk, bk, wv, bv, wo, bo):
    h = CONFIG['num_heads']
    q = Q @ wq + bq
    k = K @ wk + bk
    v = K @ wv + bv
    B, Nq, dv = q.shape
    Nk = k.shape[1]
    d = dv // h
    qh = q.reshape(B, Nq, h, d)
    kh = k.reshape(B, Nk, h, d)
    vh = v.reshape(B, Nk, h, d)
    s = jnp.einsum('bqhd,bkhd->bhqk', qh, kh) / (dv ** 0.5)
    a = jax.nn.softmax(s, axis=-1)
    o = qh + jnp.einsum('bhqk,bkhd->bqhd', a, vh)
    o = o.reshape(B, Nq, dv)
    return o + jax.nn.relu(o @ wo + bo)
