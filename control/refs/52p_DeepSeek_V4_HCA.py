import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp


def cdiv(a, b):
  assert b != 0
  return (a + b - 1) // b


CONFIGS = {
    'decode_small': {
        'name': 'decode_small',
        'model': 'DeepSeek-V4',
        'operator': 'heavily_compressed_attention',
        'batch_size': 16,
        'q_len': 1,
        'kv_len_val': 1024,
        'num_q_heads': 128,
        'head_dim': 512,
        'page_size': 256,
        'compress_ratio': 128,
        'sm_scale': 1.0,
        'num_kv_pages_per_block': 1,
        'num_queries_per_block': 1,
    },
    'dsv4_pro_decode': {
        'name': 'dsv4_pro_decode',
        'model': 'DeepSeek-V4-Pro',
        'operator': 'heavily_compressed_attention',
        'batch_size': 256,
        'q_len': 1,
        'kv_len_val': 9216,
        'num_q_heads': 128,
        'head_dim': 512,
        'page_size': 1024,
        'compress_ratio': 128,
        'sm_scale': 1.0,
        'num_kv_pages_per_block': 16,
        'num_queries_per_block': 1,
    },
    'dsv4_pro_prefill_chunk_first': {
        'name': 'dsv4_pro_prefill_chunk_first',
        'model': 'DeepSeek-V4-Pro',
        'operator': 'heavily_compressed_attention',
        'batch_size': 1,
        'q_len': 1024,
        'kv_len_val': 1024,
        'num_q_heads': 128,
        'head_dim': 512,
        'page_size': 1024,
        'compress_ratio': 128,
        'sm_scale': 1.0,
        'num_kv_pages_per_block': 16,
        'num_queries_per_block': 32,
    },
    'dsv4_pro_prefill_chunk_last': {
        'name': 'dsv4_pro_prefill_chunk_last',
        'model': 'DeepSeek-V4-Pro',
        'operator': 'heavily_compressed_attention',
        'batch_size': 1,
        'q_len': 1024,
        'kv_len_val': 8192,
        'num_q_heads': 128,
        'head_dim': 512,
        'page_size': 1024,
        'compress_ratio': 128,
        'sm_scale': 1.0,
        'num_kv_pages_per_block': 16,
        'num_queries_per_block': 32,
    },
    'dsv4_flash_decode': {
        'name': 'dsv4_flash_decode',
        'model': 'DeepSeek-V4-Flash',
        'operator': 'heavily_compressed_attention',
        'batch_size': 256,
        'q_len': 1,
        'kv_len_val': 9216,
        'num_q_heads': 64,
        'head_dim': 512,
        'page_size': 1024,
        'compress_ratio': 128,
        'sm_scale': 1.0,
        'num_kv_pages_per_block': 16,
        'num_queries_per_block': 1,
    },
    'dsv4_flash_prefill_chunk_last': {
        'name': 'dsv4_flash_prefill_chunk_last',
        'model': 'DeepSeek-V4-Flash',
        'operator': 'heavily_compressed_attention',
        'batch_size': 1,
        'q_len': 1024,
        'kv_len_val': 8192,
        'num_q_heads': 64,
        'head_dim': 512,
        'page_size': 1024,
        'compress_ratio': 128,
        'sm_scale': 1.0,
        'num_kv_pages_per_block': 16,
        'num_queries_per_block': 32,
    },
}


CONFIG = CONFIGS['decode_small']


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns input tensors for DeepSeek V4 HCA."""
  cfg = (
      CONFIG
      if config is None
      else (CONFIGS[config] if isinstance(config, str) else config)
  )

  key = jax.random.key(42)
  k1, k2, k3, k4, k5 = jax.random.split(key, 5)

  B = cfg['batch_size']
  q_len = cfg['q_len']
  kv_len_val = cfg['kv_len_val']
  num_heads = cfg['num_q_heads']
  head_dim = cfg['head_dim']
  raw_page_size = cfg['page_size']
  compress_ratio = cfg.get('compress_ratio', 128)

  page_size = max(1, raw_page_size // compress_ratio)
  kv_len_val_after_compression = max(1, kv_len_val // compress_ratio)
  num_tokens = B * q_len

  pages_per_seq = cdiv(kv_len_val_after_compression, page_size) + 2
  total_pages = B * pages_per_seq

  q = jax.random.normal(k1, (num_tokens, num_heads, head_dim), dtype=dtype)

  # Cache format: [total_pages, page_size * 2, 4, 128] uint8 bitcasted from bf16
  cache_kv_base = jax.random.normal(
      k2, (total_pages, page_size, head_dim), dtype=jnp.bfloat16
  )
  k_cache = cache_kv_base.reshape(total_pages, page_size, 4, 128)
  k_cache = jax.lax.bitcast_convert_type(k_cache, jnp.uint8)
  k_cache = k_cache.transpose(0, 1, 2, 4, 3)
  kernel_cache = k_cache.reshape(total_pages, page_size * 2, 4, 128)

  kv_lens = jnp.full((B,), kv_len_val_after_compression, dtype=jnp.int32)
  cu_q_lens = jnp.arange(0, num_tokens + 1, q_len, dtype=jnp.int32)
  page_indices = jnp.arange(total_pages, dtype=jnp.int32)

  num_decode_seqs = B if q_len == 1 else 0
  distribution = jnp.array(
      [num_decode_seqs, num_decode_seqs, B], dtype=jnp.int32
  )

  attention_sinks = jax.random.uniform(k3, (num_heads,), dtype=jnp.float32)
  swa_accumulation = jax.random.normal(
      k4, (num_tokens, num_heads, head_dim), dtype=dtype
  )
  swa_l = jax.random.uniform(k5, (num_tokens, num_heads), dtype=jnp.float32)
  swa_m = jax.random.uniform(k1, (num_tokens, num_heads), dtype=jnp.float32)

  kv_lens_to_attend_list = []
  for _ in range(B):
    for i in range(q_len):
      kv_lens_to_attend_list.append(
          (kv_len_val - q_len + i) // compress_ratio
      )
  kv_lens_to_attend = jnp.array(kv_lens_to_attend_list, dtype=jnp.int32)

  return (
      q,
      kernel_cache,
      kv_lens,
      kv_lens_to_attend,
      page_indices,
      cu_q_lens,
      distribution,
      attention_sinks,
      swa_accumulation,
      swa_l,
      swa_m,
  )


import jax
import jax.numpy as jnp


def workload(
    q,
    cache_kv,
    kv_lens,
    kv_lens_to_attend,
    page_indices,
    cu_q_lens,
    distribution,
    attention_sinks,
    swa_accumulation,
    swa_l,
    swa_m,
):
  """Plain JAX implementation of DeepSeek V4 HCA ragged paged attention."""
  num_tokens, num_heads, head_dim = q.shape
  total_pages, num_slots, kv_packing, lkv_dim = cache_kv.shape
  page_size = num_slots // 2
  max_num_seqs = kv_lens.shape[0]
  pages_per_seq = page_indices.shape[0] // max_num_seqs
  max_kv_len = pages_per_seq * page_size
  sm_scale = 1.0

  # Unpack cache_kv back to bfloat16 of shape [total_pages, page_size, head_dim].
  # In the Pallas kernel, each KV token is stored across 2 slots of [kv_packing, lkv_dim]
  # uint8, which inside VMEM is reshaped to [2 * kv_packing, lkv_dim] uint8 and
  # bitcast on TPU (where sub-32-bit bitcasts pack along the second-to-last axis)
  # to [kv_packing, lkv_dim] bfloat16, then flattened to [head_dim].
  cache_u8 = cache_kv.reshape(total_pages, page_size, 2, kv_packing, lkv_dim)
  cache_u8 = cache_u8.transpose(0, 1, 3, 4, 2)
  cache_bf16 = jax.lax.bitcast_convert_type(cache_u8, jnp.bfloat16).reshape(
      total_pages, page_size, head_dim
  )

  # Gather sequence KV caches: [max_num_seqs, max_kv_len, head_dim]
  seq_page_indices = page_indices.reshape(max_num_seqs, pages_per_seq)
  seq_kv = cache_bf16[seq_page_indices].reshape(
      max_num_seqs, max_kv_len, head_dim
  )

  # Map each query token to its sequence index using cu_q_lens.
  num_active_seqs = distribution[2]
  token_idx = jnp.arange(num_tokens, dtype=jnp.int32)
  seq_ids = (
      jnp.searchsorted(cu_q_lens[1:], token_idx, side='right')
      .astype(jnp.int32)
      .clip(0, max_num_seqs - 1)
  )
  valid_token = (token_idx < cu_q_lens[num_active_seqs]) & (
      seq_ids < num_active_seqs
  )

  # Effective KV length each token attends to (bounded by both kv_lens_to_attend and kv_lens).
  eff_kv_len = jnp.minimum(kv_lens_to_attend, kv_lens[seq_ids])
  k_pos = jnp.arange(max_kv_len, dtype=jnp.int32)

  # Choose the fastest path based on batch / token dimensions.
  if num_tokens == max_num_seqs:
    # 1 token per sequence (e.g., decode): avoid gathering seq_kv by token_idx
    # if seq_ids is the identity permutation, or gather directly.
    kv_for_t = seq_kv[seq_ids]  # [T, K, D]
    s = (
        jnp.einsum(
            'thd,tkd->thk',
            q,
            kv_for_t,
            preferred_element_type=jnp.float32,
        )
        * sm_scale
    )
    mask = k_pos[None, None, :] >= eff_kv_len[:, None, None]
    fmin = jnp.finfo(jnp.float32).min
    s = jnp.where(mask, fmin, s)

    s_max = jnp.max(s, axis=-1)
    m_curr = jnp.maximum(swa_m, s_max)
    p = jnp.exp(s - m_curr[:, :, None])
    pv = jnp.einsum(
        'thk,tkd->thd',
        p,
        kv_for_t,
        preferred_element_type=jnp.float32,
    )
    p_sum = jnp.sum(p, axis=-1)
  elif max_num_seqs == 1:
    # Single sequence prefill: avoid replicating KV across tokens.
    kv_single = seq_kv[0]  # [K, D]
    s = (
        jnp.einsum(
            'thd,kd->thk',
            q,
            kv_single,
            preferred_element_type=jnp.float32,
        )
        * sm_scale
    )
    mask = k_pos[None, None, :] >= eff_kv_len[:, None, None]
    fmin = jnp.finfo(jnp.float32).min
    s = jnp.where(mask, fmin, s)

    s_max = jnp.max(s, axis=-1)
    m_curr = jnp.maximum(swa_m, s_max)
    p = jnp.exp(s - m_curr[:, :, None])
    pv = jnp.einsum(
        'thk,kd->thd',
        p,
        kv_single,
        preferred_element_type=jnp.float32,
    )
    p_sum = jnp.sum(p, axis=-1)
  else:
    # General ragged case.
    kv_for_t = seq_kv[seq_ids]  # [T, K, D]
    s = (
        jnp.einsum(
            'thd,tkd->thk',
            q,
            kv_for_t,
            preferred_element_type=jnp.float32,
        )
        * sm_scale
    )
    mask = k_pos[None, None, :] >= eff_kv_len[:, None, None]
    fmin = jnp.finfo(jnp.float32).min
    s = jnp.where(mask, fmin, s)

    s_max = jnp.max(s, axis=-1)
    m_curr = jnp.maximum(swa_m, s_max)
    p = jnp.exp(s - m_curr[:, :, None])
    pv = jnp.einsum(
        'thk,tkd->thd',
        p,
        kv_for_t,
        preferred_element_type=jnp.float32,
    )
    p_sum = jnp.sum(p, axis=-1)

  # Combine with SWA state and attention sinks.
  exp_m_diff = jnp.exp(swa_m - m_curr)
  l_curr = exp_m_diff * swa_l + p_sum
  acc = exp_m_diff[:, :, None] * swa_accumulation.astype(jnp.float32) + pv

  exp_sinks = jnp.exp(attention_sinks[None, :] - m_curr)
  l_final = l_curr + exp_sinks
  out = (acc / l_final[:, :, None]).astype(q.dtype)

  # Untouched tokens (outside active sequences) retain original q values due to input_output_aliases.
  out = jnp.where(valid_token[:, None, None], out, q)
  return out
