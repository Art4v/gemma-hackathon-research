"""[m1 M2_second_output variant of authored/attention_decode_mqa/mqa_decode_padded_cache] MQA single-token decode over a padded cache that also returns the per-head softmax logsumexp (B, H), the statistic split-KV decode uses to combine partial results."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'mqa_decode_padded_cache_m1_aux',
    'batch': 32,
    'num_q_heads': 71,
    'head_dim': 64,
    'max_seq_len': 32768,
    'min_len': 1,
    'mask_value': -1e30,
}


def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch']
    H = CONFIG['num_q_heads']
    D = CONFIG['head_dim']
    S = CONFIG['max_seq_len']
    key = jax.random.key(0)
    kq, kk, kv, kl = jax.random.split(key, 4)
    q = jax.random.normal(kq, (B, H, D), dtype)
    k_cache = jax.random.normal(kk, (B, S, D), dtype)
    v_cache = jax.random.normal(kv, (B, S, D), dtype)
    lengths = jax.random.randint(kl, (B,), CONFIG['min_len'], S + 1, dtype=jnp.int32)
    return q, k_cache, v_cache, lengths


def workload(q, k_cache, v_cache, lengths):
    D = q.shape[-1]
    S = k_cache.shape[1]
    scale = 1.0 / (D ** 0.5)
    qf = q.astype(jnp.float32)
    kf = k_cache.astype(jnp.float32)
    vf = v_cache.astype(jnp.float32)
    # [B, H, S] scores; every query head attends to the single shared KV head.
    scores = jnp.einsum('bhd,bsd->bhs', qf, kf) * scale
    pos = jnp.arange(S, dtype=jnp.int32)
    valid = pos[None, :] < lengths[:, None]  # [B, S]
    scores = jnp.where(valid[:, None, :], scores, CONFIG['mask_value'])
    m = jnp.max(scores, axis=-1, keepdims=True)
    p = jnp.exp(scores - m)
    p = jnp.where(valid[:, None, :], p, 0.0)
    denom = jnp.sum(p, axis=-1, keepdims=True)
    out = jnp.einsum('bhs,bsd->bhd', p, vf) / denom
    lse = (m + jnp.log(denom))[..., 0]  # [B, H] f32; lengths >= 1 so denom > 0
    return out.astype(q.dtype), lse
