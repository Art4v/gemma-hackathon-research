"""Mixed local (sliding-window) / global causal multi-head attention with logit softcapping: QK^T -> softcap -> per-head window/causal mask -> softmax -> PV."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'local_global_head_attention',
    'batch': 1,
    'num_heads': 32,
    'seq_len': 4096,
    'head_dim': 128,
    'window': 1024,          # sliding window (keys incl. self) for local heads
    'local_head_period': 2,  # head h is local iff h % period == 0
    'softcap': 50.0,
    'scale': 0.08838834764831845,  # 1/sqrt(head_dim)
    'q_chunk': 512,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kq, kk, kv = jax.random.split(key, 3)
    shape = (CONFIG['batch'], CONFIG['num_heads'], CONFIG['seq_len'], CONFIG['head_dim'])
    q = jax.random.normal(kq, shape, dtype)
    k = jax.random.normal(kk, shape, dtype)
    v = jax.random.normal(kv, shape, dtype)
    return q, k, v


def workload(q, k, v):
    B, H, S, D = q.shape
    chunk = CONFIG['q_chunk']
    n_chunks = S // chunk
    scale = CONFIG['scale']
    cap = CONFIG['softcap']

    head_idx = jnp.arange(H)
    is_local = (head_idx % CONFIG['local_head_period']) == 0
    window = jnp.where(is_local, CONFIG['window'], S)  # (H,)

    q_blocks = q.reshape(B, H, n_chunks, chunk, D).transpose(2, 0, 1, 3, 4)
    starts = jnp.arange(n_chunks, dtype=jnp.int32) * chunk
    kpos = jnp.arange(S, dtype=jnp.int32)

    def body(args):
        qblk, start = args
        s = jnp.einsum('bhqd,bhkd->bhqk', qblk, k,
                       preferred_element_type=jnp.float32) * scale
        s = cap * jnp.tanh(s / cap)
        qpos = start + jnp.arange(chunk, dtype=jnp.int32)
        diff = qpos[:, None] - kpos[None, :]                      # (c, S)
        mask = (diff[None] >= 0) & (diff[None] < window[:, None, None])  # (H, c, S)
        s = jnp.where(mask[None], s, -jnp.inf)
        m = jnp.max(s, axis=-1, keepdims=True)
        p = jnp.exp(s - m)
        l = jnp.sum(p, axis=-1, keepdims=True)
        o = jnp.einsum('bhqk,bhkd->bhqd', p.astype(v.dtype), v,
                       preferred_element_type=jnp.float32)
        o = o / l
        return o.astype(q.dtype)

    out = lax.map(body, (q_blocks, starts))           # (n, B, H, c, D)
    return out.transpose(1, 2, 0, 3, 4).reshape(B, H, S, D)
