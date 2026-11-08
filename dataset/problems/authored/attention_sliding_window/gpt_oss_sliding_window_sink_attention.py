"""gpt-oss sliding-window causal GQA attention with learned per-head attention sinks (banded QK^T -> masked sink-softmax in f32 -> PV)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gpt_oss_sliding_window_sink_attention',
    'batch': 1,
    'seq_len': 8192,
    'num_q_heads': 64,
    'num_kv_heads': 8,
    'head_dim': 64,
    'window': 128,
    'sink_scale': 1.0,
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
    sinks = jax.random.normal(ks, (Hq,), dtype) * CONFIG['sink_scale']
    return q, k, v, sinks


def workload(q, k, v, sinks):
    B, S, Hq, D = q.shape
    Hkv = k.shape[2]
    G = Hq // Hkv
    W = CONFIG['window']
    nb = S // W
    scale = 1.0 / jnp.sqrt(jnp.float32(D))

    # Query blocks: (B, nb, W, Hkv, G, D)
    qb = q.reshape(B, nb, W, Hkv, G, D)
    # Key/value blocks covering [previous block, current block]: (B, nb, 2W, Hkv, D)
    kb = k.reshape(B, nb, W, Hkv, D)
    vb = v.reshape(B, nb, W, Hkv, D)
    k_prev = jnp.pad(kb, ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)))[:, :nb]
    v_prev = jnp.pad(vb, ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)))[:, :nb]
    kw = jnp.concatenate([k_prev, kb], axis=2)
    vw = jnp.concatenate([v_prev, vb], axis=2)

    # Scores in f32: (B, nb, Hkv, G, W, 2W)
    s = jnp.einsum('bnqhgd,bnkhd->bnhgqk', qb, kw,
                   preferred_element_type=jnp.float32) * scale

    # Banded causal mask: q_pos - W < k_pos <= q_pos, k_pos >= 0
    blk = jnp.arange(nb)[:, None, None]
    qi = jnp.arange(W)[None, :, None]
    kj = jnp.arange(2 * W)[None, None, :]
    q_pos = blk * W + qi
    k_pos = (blk - 1) * W + kj
    diff = q_pos - k_pos
    mask = (diff >= 0) & (diff < W) & (k_pos >= 0)  # (nb, W, 2W)
    mask = mask[None, :, None, None, :, :]
    neg = jnp.finfo(jnp.float32).min
    s = jnp.where(mask, s, neg)

    # Sink-augmented softmax in f32
    sink = sinks.astype(jnp.float32).reshape(Hkv, G)[None, None, :, :, None, None]
    m = jnp.maximum(jnp.max(s, axis=-1, keepdims=True), sink)
    p = jnp.where(mask, jnp.exp(s - m), 0.0)
    denom = jnp.sum(p, axis=-1, keepdims=True) + jnp.exp(sink - m)
    p = p / denom

    o = jnp.einsum('bnhgqk,bnkhd->bnqhgd', p, vw.astype(jnp.float32),
                   preferred_element_type=jnp.float32)
    return o.reshape(B, S, Hq, D).astype(q.dtype)
