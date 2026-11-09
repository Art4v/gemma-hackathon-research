"""[bf16 variant] Stick-breaking attention (naive log-space): GQA scores -> logsigmoid stick-breaking weights -> weighted V, plus remainder mass."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'stickbreaking_attention',
    'batch': 8,
    'seq_len': 1024,
    'num_q_heads': 32,
    'num_kv_heads': 8,
    'head_dim': 128,
    'attend_current': False,
}


def create_inputs(dtype=jnp.bfloat16):
    kq, kk, kv = jax.random.split(jax.random.key(0), 3)
    B, T, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['head_dim']
    HQ, H = CONFIG['num_q_heads'], CONFIG['num_kv_heads']
    q = jax.random.normal(kq, (B, T, HQ, D), dtype)
    k = jax.random.normal(kk, (B, T, H, D), dtype)
    v = jax.random.normal(kv, (B, T, H, D), dtype)
    return q, k, v


def workload(q, k, v):
    B, T, HQ, K = q.shape
    H, V = k.shape[2], v.shape[-1]
    G = HQ // H
    scale = K ** -0.5
    dtype = q.dtype
    q, k, v = q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)
    z = jnp.einsum('bqhgd,bkhd->bhgqk', q.reshape(B, T, H, G, K), k,
                   precision=lax.Precision.HIGHEST) * scale
    i = jnp.arange(T)
    if CONFIG['attend_current']:
        mask = i[None, :] <= i[:, None]
    else:
        mask = i[None, :] < i[:, None]
    log_om_beta = jnp.where(mask, jax.nn.log_sigmoid(-z), 0.0)
    rev_cumsum = jnp.flip(jnp.cumsum(jnp.flip(log_om_beta, -1), axis=-1), -1)
    log_att = jax.nn.log_sigmoid(z) + rev_cumsum - log_om_beta
    att = jnp.exp(jnp.where(mask, log_att, -jnp.inf))
    o = jnp.einsum('bhgqk,bkhd->bqhgd', att, v,
                   precision=lax.Precision.HIGHEST).reshape(B, T, HQ, V)
    rem = jnp.exp(jnp.sum(log_om_beta, axis=-1)).transpose(0, 3, 1, 2).reshape(B, T, HQ)
    return o.astype(dtype), rem.astype(dtype)
