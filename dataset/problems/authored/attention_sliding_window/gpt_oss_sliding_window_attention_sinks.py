"""gpt-oss sliding-window causal GQA attention with learned per-head attention sinks (QK^T -> band mask -> sink-augmented f32 softmax -> PV)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gpt_oss_sliding_window_attention_sinks',
    'batch': 1,
    'seq_len': 8192,
    'num_q_heads': 64,
    'num_kv_heads': 8,
    'head_dim': 64,
    'window': 128,
}


def create_inputs(dtype=jnp.float32):
    B = CONFIG['batch']
    S = CONFIG['seq_len']
    Hq = CONFIG['num_q_heads']
    Hkv = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']
    kq, kk, kv, ks = jax.random.split(jax.random.key(0), 4)
    q = jax.random.normal(kq, (B, S, Hq, D), dtype)
    k = jax.random.normal(kk, (B, S, Hkv, D), dtype)
    v = jax.random.normal(kv, (B, S, Hkv, D), dtype)
    sinks = jax.random.normal(ks, (Hq,), dtype) * 0.02
    return q, k, v, sinks


def workload(q, k, v, sinks):
    B, S, Hq, D = q.shape
    Hkv = k.shape[2]
    G = Hq // Hkv
    W = CONFIG['window']
    nb = S // W
    scale = 1.0 / (D ** 0.5)

    # Query blocks of size W; each attends to [previous block, current block].
    qb = q.reshape(B, nb, W, Hkv, G, D)
    k_pad = jnp.pad(k, ((0, 0), (W, 0), (0, 0), (0, 0)))
    v_pad = jnp.pad(v, ((0, 0), (W, 0), (0, 0), (0, 0)))
    kk = jnp.concatenate([k_pad[:, :S].reshape(B, nb, W, Hkv, D),
                          k.reshape(B, nb, W, Hkv, D)], axis=2)  # (B,nb,2W,Hkv,D)
    vv = jnp.concatenate([v_pad[:, :S].reshape(B, nb, W, Hkv, D),
                          v.reshape(B, nb, W, Hkv, D)], axis=2)

    s = jnp.einsum('bnqhgd,bnkhd->bnhgqk', qb, kk,
                   preferred_element_type=jnp.float32) * scale  # (B,nb,Hkv,G,W,2W)

    qi = jnp.arange(W)[:, None]
    kj = jnp.arange(2 * W)[None, :]
    diff = (qi + W) - kj  # query pos minus key pos
    blk = jnp.arange(nb)[:, None, None]
    kpos = blk * W - W + kj[None]
    mask = (diff[None] >= 0) & (diff[None] < W) & (kpos >= 0)  # (nb,W,2W)
    mask = mask[None, :, None, None, :, :]

    s = jnp.where(mask, s, -jnp.inf)
    sink = sinks.astype(jnp.float32).reshape(1, 1, Hkv, G, 1, 1)
    m = jnp.maximum(jnp.max(s, axis=-1, keepdims=True), sink)
    e = jnp.where(mask, jnp.exp(s - m), 0.0)
    denom = jnp.sum(e, axis=-1, keepdims=True) + jnp.exp(sink - m)
    p = e / denom

    out = jnp.einsum('bnhgqk,bnkhd->bnqhgd', p.astype(v.dtype), vv,
                     preferred_element_type=jnp.float32)
    return out.reshape(B, S, Hq, D).astype(q.dtype)
