"""[m1 M2_second_output variant of authored/attention_chunked_prefill/chunked_prefill_attention] Chunked-prefill causal attention that also returns the per-row softmax logsumexp (B, H, C), which a scheduler uses to merge partial attention results."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'chunked_prefill_attention_m1_aux',
    'batch': 4,
    'num_heads': 32,
    'chunk_len': 512,
    'cache_len': 4096,
    'head_dim': 128,
    'scale': 128 ** -0.5,
    'mask_value': -1e30,
}


def create_inputs(dtype=jnp.float32):
    B = CONFIG['batch']
    H = CONFIG['num_heads']
    C = CONFIG['chunk_len']
    S = CONFIG['cache_len']
    D = CONFIG['head_dim']
    key = jax.random.key(0)
    kq, kk, kv, kp = jax.random.split(key, 4)
    q = jax.random.normal(kq, (B, H, C, D), dtype)
    k_cache = jax.random.normal(kk, (B, H, S, D), dtype)
    v_cache = jax.random.normal(kv, (B, H, S, D), dtype)
    prefix_lens = jax.random.randint(kp, (B,), 0, S - C + 1, dtype=jnp.int32)
    return q, k_cache, v_cache, prefix_lens


def workload(q, k_cache, v_cache, prefix_lens):
    C = q.shape[2]
    S = k_cache.shape[2]
    scale = CONFIG['scale']
    s = jnp.einsum('bhqd,bhkd->bhqk', q, k_cache,
                   preferred_element_type=jnp.float32) * scale
    q_pos = prefix_lens[:, None] + jnp.arange(C, dtype=jnp.int32)[None, :]  # [B, C]
    k_pos = jnp.arange(S, dtype=jnp.int32)  # [S]
    mask = k_pos[None, None, :] <= q_pos[:, :, None]  # [B, C, S]
    s = jnp.where(mask[:, None, :, :], s, CONFIG['mask_value'])
    m = jnp.max(s, axis=-1, keepdims=True)
    p = jnp.exp(s - m)
    l = jnp.sum(p, axis=-1, keepdims=True)
    o = jnp.einsum('bhqk,bhkd->bhqd', p.astype(v_cache.dtype), v_cache,
                   preferred_element_type=jnp.float32)
    o = o / l
    lse = (m + jnp.log(l))[..., 0]  # [B, H, C] f32
    return o.astype(q.dtype), lse
