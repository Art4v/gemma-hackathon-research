"""Chunked prefill causal GQA with per-batch query offsets, sliding window, and Gemma-2 logit soft-capping."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'chunked_prefill_gemma2_softcap_attention',
    'batch_size': 8,
    'q_len': 1024,
    'kv_len': 8192,
    'num_q_heads': 32,
    'num_kv_heads': 8,
    'head_dim': 128,
    ' sliding_window': 4096,
    'softcap': 50.0,
    'sm_scale': 0.08838834764831845,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_q, k_k, k_v, k_off = jax.random.split(key, 4)
    B = CONFIG['batch_size']
    Q_L = CONFIG['q_len']
    KV_L = CONFIG['kv_len']
    H_q = CONFIG['num_q_heads']
    H_kv = CONFIG['num_kv_heads']
    D = CONFIG['head_dim']

    q = jax.random.normal(k_q, (B, Q_L, H_q, D), dtype=dtype)
    k = jax.random.normal(k_k, (B, KV_L, H_kv, D), dtype=dtype)
    v = jax.random.normal(k_v, (B, KV_L, H_kv, D), dtype=dtype)
    prefix_lens = jax.random.randint(k_off, (B,), minval=0, maxval=KV_L - Q_L + 1, dtype=jnp.int32)
    return q, k, v, prefix_lens


def workload(q, k, v, prefix_lens):
    B, Q_L, H_q, D = q.shape
    _, KV_L, H_kv, _ = k.shape
    groups = H_q // H_kv
    sm_scale = CONFIG['sm_scale']
    softcap = CONFIG['softcap']
    window = CONFIG[' sliding_window']

    q_grouped = q.reshape(B, Q_L, H_kv, groups, D).astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    logits = jnp.einsum('bqhgd,bkhd->bhgqk', q_grouped, k_f32) * sm_scale
    logits = softcap * jnp.tanh(logits / softcap)

    q_pos = prefix_lens[:, None] + jnp.arange(Q_L, dtype=jnp.int32)[None, :]
    k_pos = jnp.arange(KV_L, dtype=jnp.int32)[None, None, :]
    q_pos_exp = q_pos[:, :, None]

    causal_mask = k_pos <= q_pos_exp
    window_mask = k_pos > (q_pos_exp - window)
    valid_mask = (causal_mask & window_mask)[:, None, None, :, :]

    logits = jnp.where(valid_mask, logits, -1e30)
    max_logits = jnp.max(logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(logits - max_logits)
    probs = exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)

    out = jnp.einsum('bhgqk,bkhd->bqhgd', probs, v_f32)
    return out.reshape(B, Q_L, H_q, D).astype(q.dtype)
