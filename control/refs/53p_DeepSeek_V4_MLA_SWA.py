import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp


def prepare_kv_inputs(kv: jax.Array):
  """Reinterprets bf16 new KV as one raw uint8 byte row per token.

  Each token's `actual_head_dim` bf16 values become `actual_head_dim * 2`
  contiguous bytes. The bytes are laid out so that, once a token row is viewed
  as ``[slots_per_token * kv_packing, 128]`` (128 = the kernel's lane width),
  every pair of consecutive sublanes holds one bf16 value in little-endian byte
  order (lo byte then hi byte).
  """
  assert kv.dtype == jnp.bfloat16
  tokens, head_dim = kv.shape
  assert head_dim % 128 == 0
  kv_u16 = jax.lax.bitcast_convert_type(kv, jnp.uint16)
  kv_hi = ((kv_u16 >> 8) & 0xFF).astype(jnp.uint8)
  kv_lo = (kv_u16 & 0xFF).astype(jnp.uint8)
  nb = head_dim // 128
  kv_hi = kv_hi.reshape(tokens, nb, 128)
  kv_lo = kv_lo.reshape(tokens, nb, 128)
  # (tokens, nb, 2, 128)
  interleaved = jnp.stack([kv_lo, kv_hi], axis=2)
  return interleaved.reshape(tokens, head_dim * 2)


CONFIGS = {
    'decode_small': {
        'name': 'decode_small',
        'model': 'DeepSeek-V4',
        'operator': 'mla_sliding_window_attention',
        'batch_size': 16,
        'q_len': 1,
        'kv_len': 1024,
        'num_q_heads': 128,
        'head_dim': 512,
        'page_size': 256,
        'sliding_window': 128,
        'sm_scale': 1.0,
    },
    'dsv4_pro_decode_b64': {
        'name': 'dsv4_pro_decode_b64',
        'model': 'DeepSeek-V4-Pro',
        'operator': 'mla_sliding_window_attention',
        'batch_size': 64,
        'q_len': 1,
        'kv_len': 8192,
        'num_q_heads': 128,
        'head_dim': 512,
        'page_size': 1024,
        'sliding_window': 128,
        'sm_scale': 1.0,
    },
    'dsv4_pro_decode_b256': {
        'name': 'dsv4_pro_decode_b256',
        'model': 'DeepSeek-V4-Pro',
        'operator': 'mla_sliding_window_attention',
        'batch_size': 256,
        'q_len': 1,
        'kv_len': 9216,
        'num_q_heads': 128,
        'head_dim': 512,
        'page_size': 1024,
        'sliding_window': 128,
        'sm_scale': 1.0,
    },
    'dsv4_pro_prefill_short': {
        'name': 'dsv4_pro_prefill_short',
        'model': 'DeepSeek-V4-Pro',
        'operator': 'mla_sliding_window_attention',
        'batch_size': 1,
        'q_len': 1024,
        'kv_len': 1024,
        'num_q_heads': 128,
        'head_dim': 512,
        'page_size': 1024,
        'sliding_window': 128,
        'sm_scale': 1.0,
    },
    'dsv4_pro_prefill_deep': {
        'name': 'dsv4_pro_prefill_deep',
        'model': 'DeepSeek-V4-Pro',
        'operator': 'mla_sliding_window_attention',
        'batch_size': 1,
        'q_len': 1024,
        'kv_len': 8192,
        'num_q_heads': 128,
        'head_dim': 512,
        'page_size': 1024,
        'sliding_window': 128,
        'sm_scale': 1.0,
    },
    'dsv4_flash_decode': {
        'name': 'dsv4_flash_decode',
        'model': 'DeepSeek-V4-Flash',
        'operator': 'mla_sliding_window_attention',
        'batch_size': 256,
        'q_len': 1,
        'kv_len': 9216,
        'num_q_heads': 64,
        'head_dim': 512,
        'page_size': 1024,
        'sliding_window': 128,
        'sm_scale': 1.0,
    },
    'dsv4_flash_prefill': {
        'name': 'dsv4_flash_prefill',
        'model': 'DeepSeek-V4-Flash',
        'operator': 'mla_sliding_window_attention',
        'batch_size': 1,
        'q_len': 1024,
        'kv_len': 8192,
        'num_q_heads': 64,
        'head_dim': 512,
        'page_size': 1024,
        'sliding_window': 128,
        'sm_scale': 1.0,
    },
}


CONFIG = CONFIGS['decode_small']


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns (q, new_kv, kernel_cache, kv_lens, page_indices, cu_q_lens, distribution, attention_sinks)."""
  cfg = (
      CONFIG
      if config is None
      else (CONFIGS[config] if isinstance(config, str) else config)
  )

  key = jax.random.key(42)
  k1, k2, k3, k4 = jax.random.split(key, 4)

  B = cfg['batch_size']
  q_len = cfg['q_len']
  kv_len = cfg['kv_len']
  num_tokens = B * q_len
  num_q_heads = cfg['num_q_heads']
  head_dim = cfg['head_dim']
  page_size = (
      cfg['page_size'] // 4 if cfg['page_size'] > 256 else cfg['page_size']
  )

  logical_page_size = 128
  pages_per_seq = (kv_len + logical_page_size - 1) // logical_page_size + 2
  total_pages = B * pages_per_seq

  q = jax.random.normal(k1, (num_tokens, num_q_heads, head_dim), dtype=dtype)
  new_kv = jax.random.normal(k2, (num_tokens, head_dim), dtype=jnp.bfloat16)
  total_tokens = total_pages * logical_page_size
  cache_tokens = jax.random.normal(k3, (total_tokens, head_dim), dtype=jnp.bfloat16)
  kernel_cache = prepare_kv_inputs(cache_tokens).reshape(
      total_pages, page_size, 4, 128
  )

  kv_lens = jnp.full((B,), kv_len, dtype=jnp.int32)
  page_indices = jnp.arange(total_pages, dtype=jnp.int32)
  cu_q_lens = jnp.arange(0, num_tokens + 1, q_len, dtype=jnp.int32)
  num_decode_seqs = B if q_len == 1 else 0
  distribution = jnp.array(
      [num_decode_seqs, num_decode_seqs, B], dtype=jnp.int32
  )
  attention_sinks = jax.random.uniform(k4, (num_q_heads,), dtype=jnp.float32)

  return (
      q,
      new_kv,
      kernel_cache,
      kv_lens,
      page_indices,
      cu_q_lens,
      distribution,
      attention_sinks,
  )


import jax
import jax.numpy as jnp


def _unpack_kv_cache(kernel_cache: jax.Array, head_dim: int) -> jax.Array:
  """Unpacks the raw uint8 byte layout back into bfloat16 tokens."""
  total_pages, phys_page_sz_per_pack, kv_packing, lkv_dim = kernel_cache.shape
  token_bytes = head_dim * 2
  slot_bytes = kv_packing * lkv_dim
  slots_per_token = token_bytes // slot_bytes
  phys_tokens_per_page = phys_page_sz_per_pack // slots_per_token
  nb = head_dim // 128

  u8 = kernel_cache.reshape(total_pages * phys_tokens_per_page, nb, 2, 128)
  lo = u8[:, :, 0, :].astype(jnp.uint16)
  hi = u8[:, :, 1, :].astype(jnp.uint16)
  u16 = lo | (hi << 8)
  bf16 = jax.lax.bitcast_convert_type(u16, jnp.bfloat16)
  return bf16.reshape(total_pages * phys_tokens_per_page, head_dim)


def workload(
    q,
    new_kv,
    kernel_cache,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    attention_sinks,
):
  """Plain JAX implementation of MLA Sliding Window Ragged Paged Attention."""
  del attention_sinks  # Unused when unnormalized_output=True
  num_tokens, num_q_heads, head_dim = q.shape
  max_num_seqs = kv_lens.shape[0]
  pages_per_seq = page_indices.shape[0] // max_num_seqs

  sliding_window = 128
  sm_scale = 1.0
  logical_page_size = 128

  total_pages, phys_page_sz_per_pack, kv_packing, lkv_dim = kernel_cache.shape
  slot_bytes = kv_packing * lkv_dim
  slots_per_token = (head_dim * 2) // slot_bytes
  phys_tokens_per_page = phys_page_sz_per_pack // slots_per_token

  cache_bf16 = _unpack_kv_cache(kernel_cache, head_dim)
  page_table = page_indices.reshape(max_num_seqs, pages_per_seq)

  num_valid_seqs = distribution[2]
  total_valid_tokens = cu_q_lens[num_valid_seqs]

  token_idx = jnp.arange(num_tokens, dtype=jnp.int32)
  # Map each token index to its corresponding sequence index.
  seq_idx = jnp.searchsorted(cu_q_lens[1:], token_idx, side='right')
  seq_idx = jnp.minimum(seq_idx, max_num_seqs - 1)

  q_start = cu_q_lens[seq_idx]
  q_end = cu_q_lens[seq_idx + 1]
  q_len = q_end - q_start
  kv_len = kv_lens[seq_idx]

  # Position of each query token within its sequence's full KV history.
  q_pos = kv_len - q_len + (token_idx - q_start)
  cache_len = kv_len - q_len

  # Each query at q_pos attends to keys at k_pos in [q_pos - sliding_window + 1, q_pos].
  w_offsets = jnp.arange(1 - sliding_window, 1, dtype=jnp.int32)
  k_pos = q_pos[:, None] + w_offsets[None, :]  # [num_tokens, sliding_window]
  valid_k = (k_pos >= 0) & (k_pos < kv_len[:, None])

  # Determine whether each key comes from the paged KV cache or from new_kv.
  from_new = k_pos >= cache_len[:, None]

  # 1. Gather indices for paged KV cache
  k_pos_clamped = jnp.clip(
      k_pos, 0, pages_per_seq * logical_page_size - 1
  )
  log_page = k_pos_clamped // logical_page_size
  in_page_off = k_pos_clamped - log_page * logical_page_size
  phys_page = page_table[seq_idx[:, None], log_page]
  cache_flat_idx = jnp.clip(
      phys_page * phys_tokens_per_page + in_page_off,
      0,
      cache_bf16.shape[0] - 1,
  )

  # 2. Gather indices for new_kv
  new_idx = jnp.clip(
      q_start[:, None] + (k_pos - cache_len[:, None]),
      0,
      num_tokens - 1,
  )

  # Process tokens in chunks to keep peak memory small on 32GB TPU v6e.
  chunk_size = min(num_tokens, 64)
  num_chunks = num_tokens // chunk_size

  q_chunks = q.reshape(num_chunks, chunk_size, num_q_heads, head_dim)
  cache_idx_chunks = cache_flat_idx.reshape(
      num_chunks, chunk_size, sliding_window
  )
  new_idx_chunks = new_idx.reshape(num_chunks, chunk_size, sliding_window)
  from_new_chunks = from_new.reshape(num_chunks, chunk_size, sliding_window)
  valid_k_chunks = valid_k.reshape(num_chunks, chunk_size, sliding_window)

  fmin = jnp.finfo(jnp.float32).min

  def compute_chunk(q_c, c_idx_c, n_idx_c, fn_c, vk_c):
    kv_from_cache = cache_bf16[c_idx_c]  # [chunk_size, W, head_dim]
    kv_from_new = new_kv[n_idx_c]        # [chunk_size, W, head_dim]
    kv_c = jnp.where(fn_c[:, :, None], kv_from_new, kv_from_cache)
    kv_c = jnp.where(vk_c[:, :, None], kv_c, jnp.bfloat16(0.0))

    scores = jnp.einsum(
        'bhd,bwd->bhw', q_c, kv_c, preferred_element_type=jnp.float32
    )
    if sm_scale != 1.0:
      scores = scores * sm_scale

    # Ensure at least one element per row is unmasked when kv_len == 0
    # to match the kernel's flash attention fallback behavior.
    fallback_keep = jnp.arange(sliding_window) == (sliding_window - 1)
    has_any_valid = jnp.any(vk_c, axis=-1, keepdims=True)
    keep = jnp.where(has_any_valid, vk_c, fallback_keep[None, :])

    scores = jnp.where(keep[:, None, :], scores, fmin)
    m_c = jnp.max(scores, axis=-1)  # [chunk_size, num_q_heads]
    p = jnp.exp(scores - m_c[:, :, None])
    l_c = jnp.sum(p, axis=-1)       # [chunk_size, num_q_heads]

    out_c = jnp.einsum(
        'bhw,bwd->bhd', p, kv_c, preferred_element_type=jnp.float32
    ).astype(q.dtype)
    return out_c, l_c, m_c

  out_chunks, l_chunks, m_chunks = jax.lax.map(
      lambda args: compute_chunk(*args),
      (
          q_chunks,
          cache_idx_chunks,
          new_idx_chunks,
          from_new_chunks,
          valid_k_chunks,
      ),
  )

  output = out_chunks.reshape(num_tokens, num_q_heads, head_dim)
  out_l = l_chunks.reshape(num_tokens, num_q_heads)
  out_m = m_chunks.reshape(num_tokens, num_q_heads)

  # Zero out any padding tokens beyond total_valid_tokens.
  valid_tok = (token_idx < total_valid_tokens) & (seq_idx < num_valid_seqs)
  output = jnp.where(valid_tok[:, None, None], output, 0)
  out_l = jnp.where(valid_tok[:, None], out_l, 0.0)
  out_m = jnp.where(valid_tok[:, None], out_m, 0.0)

  return output, out_l, out_m
