"""Chunked prefill causal attention with per-sequence prefix offsets into a KV cache."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'chunked_prefill_causal_attention',
    'batch_size': 8,
    'num_q_heads': 32,
    'num_kv_heads': 8,
    'chunk_len': 512,
    'max_kv_len': 8192,
    'head_dim': 128,
    'sm_scale': 0.08838834764831845,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_q, k_k, k_v, k_len = jax.random.split(key, 4)
    B = CONFIG['batch_size']
    H_q = CONFIG['num_q_heads']
    H_kv = CONFIG['num_kv_heads']
    S_q = CONFIG['chunk_len']
    S_kv = CONFIG['max_kv_len']
    D = CONFIG['head_dim']

    q = jax.random.normal(k_q, (B, H_q, S_q, D), dtype=dtype)
    k = jax.random.normal(k_k, (B, H_kv, S_kv, D), dtype=dtype)
    v = jax.random.normal(k_v, (B, H_kv, S_kv, D), dtype=dtype)
    prefix_lens = jax.random.randint(
        k_len, (B,), minval=S_q, maxval=S_kv - S_q, dtype=jnp.int32
    )
    return q, k, v, prefix_lens


def workload(q, k, v, prefix_lens):
    B, H_q, S_q, D = q.shape
    _, H_kv, S_kv, _ = k.shape
    groups = H_q // H_kv
    sm_scale = CONFIG['sm_scale']

    q_grouped = q.reshape(B, H_kv, groups, S_q, D).astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    v_f32 = v.astype(jnp.float32)

    logits = jnp.einsum('bhgqd,bhkd->bhgqk', q_grouped, k_f32) * sm_scale

    q_idx = jnp.arange(S_q, dtype=jnp.int32)[None, :, None]
    k_idx = jnp.arange(S_kv, dtype=jnp.int32)[None, None, :]
    q_pos = prefix_lens[:, None, None] + q_idx
    mask = k_idx <= q_pos
    mask = mask[:, None, None, :, :]

    logits = jnp.where(mask, logits, -1e30)
    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum('bhgqk,bhkd->bhgqd', weights, v_f32)
    return out.reshape(B, H_q, S_q, D).astype(q.dtype)
