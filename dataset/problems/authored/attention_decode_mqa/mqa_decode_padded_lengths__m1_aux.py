"""[m1 M2_second_output variant of authored/attention_decode_mqa/mqa_decode_padded_lengths] Single-token MQA decode over a padded shared-KV cache that also returns the per-head softmax logsumexp (B, H) in f32."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'mqa_decode_padded_lengths_m1_aux',
    'batch': 32,
    'num_q_heads': 32,
    'head_dim': 128,
    'max_seq_len': 16384,
    'sm_scale': 128 ** -0.5,
    'mask_value': -1e30,
}


def create_inputs(dtype=jnp.float32):
    B = CONFIG['batch']
    H = CONFIG['num_q_heads']
    D = CONFIG['head_dim']
    S = CONFIG['max_seq_len']
    key = jax.random.key(0)
    kq, kk, kv, kl = jax.random.split(key, 4)
    q = jax.random.normal(kq, (B, H, D), dtype)
    k_cache = jax.random.normal(kk, (B, S, D), dtype)
    v_cache = jax.random.normal(kv, (B, S, D), dtype)
    lengths = jax.random.randint(kl, (B,), 1, S + 1, dtype=jnp.int32)
    return q, k_cache, v_cache, lengths


def workload(q, k_cache, v_cache, lengths):
    S = k_cache.shape[1]
    scale = CONFIG['sm_scale']
    # Scores: every query head attends to the single shared KV head. Accumulate in f32.
    scores = jnp.einsum('bhd,bsd->bhs', q, k_cache,
                        preferred_element_type=jnp.float32) * scale
    pos = jnp.arange(S, dtype=jnp.int32)
    valid = pos[None, None, :] < lengths[:, None, None]
    scores = jnp.where(valid, scores, jnp.float32(CONFIG['mask_value']))
    m = jnp.max(scores, axis=-1, keepdims=True)
    p = jnp.exp(scores - m)
    p = jnp.where(valid, p, 0.0)
    denom = jnp.sum(p, axis=-1, keepdims=True)
    out = jnp.einsum('bhs,bsd->bhd', p, v_cache.astype(jnp.float32),
                     preferred_element_type=jnp.float32)
    out = out / denom
    lse = (m + jnp.log(denom))[..., 0]  # [B, H] f32; lengths >= 1 so denom > 0
    return out.astype(q.dtype), lse
