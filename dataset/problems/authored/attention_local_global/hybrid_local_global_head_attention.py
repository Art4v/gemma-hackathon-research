"""Hybrid per-head local (sliding-window) / global causal attention with logit soft-capping."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'hybrid_local_global_head_attention',
    'batch': 1,
    'seq_len': 4096,
    'num_heads': 32,
    'head_dim': 128,
    'window_size': 1024,
    'softcap': 50.0,
    'q_chunk': 512,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kq, kk, kv = jax.random.split(key, 3)
    shape = (CONFIG['batch'], CONFIG['seq_len'], CONFIG['num_heads'], CONFIG['head_dim'])
    q = jax.random.normal(kq, shape, dtype)
    k = jax.random.normal(kk, shape, dtype)
    v = jax.random.normal(kv, shape, dtype)
    return q, k, v


def workload(q, k, v):
    B, S, H, D = q.shape
    W = CONFIG['window_size']
    cap = CONFIG['softcap']
    qc = CONFIG['q_chunk']
    nc = S // qc
    scale = 1.0 / (D ** 0.5)

    # Even heads: local sliding window; odd heads: global causal.
    is_local = (jnp.arange(H) % 2) == 0  # (H,)
    kpos = jnp.arange(S)

    q_chunks = q.reshape(B, nc, qc, H, D).transpose(1, 0, 2, 3, 4)  # (nc,B,qc,H,D)

    def chunk_fn(args):
        qb, ci = args
        s = jnp.einsum('bqhd,bkhd->bhqk', qb, k,
                       preferred_element_type=jnp.float32) * scale
        s = cap * jnp.tanh(s / cap)
        qpos = ci * qc + jnp.arange(qc)
        causal = kpos[None, :] <= qpos[:, None]                 # (qc,S)
        in_window = (qpos[:, None] - kpos[None, :]) < W         # (qc,S)
        mask = causal[None] & (in_window[None] | ~is_local[:, None, None])  # (H,qc,S)
        s = jnp.where(mask[None], s, jnp.finfo(jnp.float32).min)
        m = jnp.max(s, axis=-1, keepdims=True)
        p = jnp.exp(s - m)
        l = jnp.sum(p, axis=-1, keepdims=True)
        p = p / l
        o = jnp.einsum('bhqk,bkhd->bqhd', p.astype(v.dtype), v,
                       preferred_element_type=jnp.float32)
        return o.astype(q.dtype)

    out = lax.map(chunk_fn, (q_chunks, jnp.arange(nc)))  # (nc,B,qc,H,D)
    return out.transpose(1, 0, 2, 3, 4).reshape(B, S, H, D)
