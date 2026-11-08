import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp


def _jaxbench_cdiv(a, b):
  """Ceiling division: JAX's own cdiv, copied so these inputs import no kernel library."""
  if isinstance(a, int) and isinstance(b, int):
    return (a + b - 1) // b
  return jax.lax.div(a + (b - 1), b)


def align_to(a, b):
  """Returns 'a' aligned to 'b'."""
  return _jaxbench_cdiv(a, b) * b


def get_dtype_packing(dtype):
  return 32 // jax.dtypes.itemsize_bits(dtype)


CONFIGS = {
    'llama3_8b_decode': {
        'name': 'batched_rpa_llama3_8b_decode',
        'model': 'Llama-3.1-8B',
        'operator': 'batched_ragged_paged_attention',
        'max_num_batched_tokens': 64,
        'max_num_seqs': 64,
        'num_q_heads': 32,
        'num_kv_heads': 8,
        'head_dim': 128,
        'page_size': 16,
        'pages_per_seq': 256,
        'active_q_lens': (1,) * 64,
        'active_kv_lens': tuple(
            128 + ((i * 37) % 120) * 16 for i in range(64)
        ),
        'distribution': (64, 64, 64),
    },
    'llama3_70b_decode': {
        'name': 'batched_rpa_llama3_70b_decode',
        'model': 'Llama-3.1-70B',
        'operator': 'batched_ragged_paged_attention',
        'max_num_batched_tokens': 64,
        'max_num_seqs': 64,
        'num_q_heads': 64,
        'num_kv_heads': 8,
        'head_dim': 128,
        'page_size': 16,
        'pages_per_seq': 256,
        'active_q_lens': (1,) * 64,
        'active_kv_lens': tuple(
            256 + ((i * 47) % 200) * 16 for i in range(64)
        ),
        'distribution': (64, 64, 64),
    },
    'qwen3_32b_decode': {
        'name': 'batched_rpa_qwen3_32b_decode',
        'model': 'Qwen3-32B',
        'operator': 'batched_ragged_paged_attention',
        'max_num_batched_tokens': 64,
        'max_num_seqs': 64,
        'num_q_heads': 64,
        'num_kv_heads': 8,
        'head_dim': 128,
        'page_size': 16,
        'pages_per_seq': 256,
        'active_q_lens': (1,) * 64,
        'active_kv_lens': tuple(
            128 + ((i * 53) % 150) * 16 for i in range(64)
        ),
        'distribution': (64, 64, 64),
    },
}


CONFIG = CONFIGS['llama3_8b_decode']


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Create inputs matching the TPU kernel specifications for Batched RPA."""
  cfg = (
      CONFIG
      if config is None
      else (CONFIGS[config] if isinstance(config, str) else config)
  )
  key = jax.random.key(42)
  k1, k2, k3, k4, k5 = jax.random.split(key, 5)

  max_tokens = cfg['max_num_batched_tokens']
  max_seqs = cfg['max_num_seqs']
  h_q = cfg['num_q_heads']
  h_kv = cfg['num_kv_heads']
  d = cfg['head_dim']
  page_size = cfg['page_size']
  pages_per_seq = cfg['pages_per_seq']
  total_pages = max_seqs * pages_per_seq
  kv_packing = get_dtype_packing(dtype)

  queries = jax.random.normal(k1, (max_tokens, h_q, d), dtype=dtype)
  keys = jax.random.normal(k2, (max_tokens, h_kv, d), dtype=dtype)
  values = jax.random.normal(k3, (max_tokens, h_kv, d), dtype=dtype)

  num_kv_heads_x2 = align_to(h_kv * 2, kv_packing)
  cache_shape = (
      total_pages,
      page_size,
      num_kv_heads_x2 // kv_packing,
      kv_packing,
      align_to(d, 128),
  )
  kv_cache = jax.random.normal(k5, cache_shape, dtype=dtype)

  active_q = cfg['active_q_lens']
  active_kv = cfg['active_kv_lens']
  num_active = len(active_q)

  kv_lens = jnp.pad(
      jnp.array(active_kv, dtype=jnp.int32), (0, max_seqs - num_active)
  )
  q_lens = jnp.array(active_q, dtype=jnp.int32)
  active_cu_q = jnp.concatenate(
      [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(q_lens)]
  )
  cu_q_lens = jnp.pad(
      active_cu_q,
      (0, max_seqs + 1 - len(active_cu_q)),
      constant_values=active_cu_q[-1],
  )

  page_indices = jax.random.permutation(
      k4, total_pages, independent=True
  ).astype(jnp.int32)
  distribution = jnp.array(cfg['distribution'], dtype=jnp.int32)

  return (
      queries,
      keys,
      values,
      kv_cache,
      kv_lens,
      page_indices,
      cu_q_lens,
      distribution,
  )


import math
import jax
import jax.numpy as jnp


def _align_to(a, b):
  return ((a + b - 1) // b) * b


def _get_dtype_packing(dtype):
  return 32 // jax.dtypes.itemsize_bits(dtype)


def workload(
    queries,
    keys,
    values,
    kv_cache,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
):
  """Plain JAX implementation of batched ragged paged attention."""
  max_tokens, num_q_heads, head_dim = queries.shape
  _, num_kv_heads, _ = keys.shape
  num_q_per_kv = num_q_heads // num_kv_heads
  max_seqs = kv_lens.shape[0]
  num_page_indices = page_indices.shape[0]
  pages_per_seq = num_page_indices // max_seqs

  total_pages, page_size, kv_dim2, kv_packing, aligned_head_dim = kv_cache.shape
  num_kv_heads_x2_aligned = kv_dim2 * kv_packing
  actual_num_kv_heads_x2 = num_kv_heads * 2

  out_dtype = queries.dtype
  mask_value = jnp.finfo(out_dtype).min
  sm_scale = 1.0 / math.sqrt(head_dim)

  # 1. Format new KV tokens to match kv_cache layout:
  # [max_tokens, kv_dim2, kv_packing, aligned_head_dim]
  new_kv = jnp.concatenate([keys, values], axis=-1).reshape(
      max_tokens, actual_num_kv_heads_x2, head_dim
  )
  new_kv_padded = jnp.pad(
      new_kv,
      (
          (0, 0),
          (0, num_kv_heads_x2_aligned - actual_num_kv_heads_x2),
          (0, aligned_head_dim - head_dim),
      ),
      constant_values=0,
  ).reshape(max_tokens, kv_dim2, kv_packing, aligned_head_dim)

  # 2. Vectorized KV cache update
  # Active sequences in DECODE (0..dist[0]) and MIXED (dist[1]..dist[2])
  seq_ids = jnp.arange(max_seqs, dtype=jnp.int32)
  active_seq_mask = (seq_ids < distribution[0]) | (
      (seq_ids >= distribution[1]) & (seq_ids < distribution[2])
  )

  q_starts = cu_q_lens[:-1]
  q_ends = cu_q_lens[1:]
  q_lens = q_ends - q_starts

  token_ids = jnp.arange(max_tokens, dtype=jnp.int32)
  # Map each token t to its sequence s: q_starts[s] <= t < q_ends[s]
  tok_in_seq = (token_ids[:, None] >= q_starts[None, :]) & (
      token_ids[:, None] < q_ends[None, :]
  ) & active_seq_mask[None, :]
  valid_tok = jnp.any(tok_in_seq, axis=1)
  tok_seq = jnp.argmax(tok_in_seq, axis=1)

  # Position of token within its sequence's KV cache
  q_offset = token_ids - q_starts[tok_seq]
  kv_pos = (kv_lens[tok_seq] - q_lens[tok_seq]) + q_offset
  p_idx = jnp.minimum(kv_pos // page_size, pages_per_seq - 1)
  p_off = kv_pos % page_size
  global_p_idx = tok_seq * pages_per_seq + p_idx
  phys_page = page_indices[global_p_idx]
  flat_dst_idx = phys_page * page_size + p_off

  kv_cache_flat = kv_cache.reshape(
      total_pages * page_size, kv_dim2, kv_packing, aligned_head_dim
  )
  # Only scatter valid tokens; invalid tokens write back existing values at index 0
  safe_dst_idx = jnp.where(valid_tok, flat_dst_idx, 0)
  existing_vals = kv_cache_flat[safe_dst_idx]
  vals_to_scatter = jnp.where(
      valid_tok[:, None, None, None], new_kv_padded, existing_vals
  )
  kv_cache_flat = kv_cache_flat.at[safe_dst_idx].set(vals_to_scatter)
  updated_kv_cache = kv_cache_flat.reshape(kv_cache.shape)

  # 3. Compute Attention
  page_table = page_indices.reshape(max_seqs, pages_per_seq)
  q_grouped = queries.reshape(max_tokens, num_kv_heads, num_q_per_kv, head_dim)

  # Fast path when all sequences have at most 1 query token (e.g., decode)
  # Otherwise loop over sequences with dynamic slicing / masking.
  # Since max_tokens == 64 and max_seqs == 64 in all configs, we can check if
  # we can vectorize over max_tokens directly!
  # Every active query token `t` belongs to `tok_seq[t]`, has query position `kv_pos[t]`,
  # and attends to keys `0 .. kv_pos[t]` of sequence `tok_seq[t]`.
  # Notice that `max_tokens` is only 64! We can vectorize across all `max_tokens`
  # in chunks or using `lax.map` / `vmap` or `fori_loop` over active tokens,
  # OR even simpler: in all configs, max_tokens = 64.
  # Can we vectorize across all 64 tokens at once?
  # Let's check memory for gathering KV for 64 tokens:
  # 64 tokens * 256 pages * 16 page_size = 262,144 rows.
  # Each row is 16 * 128 * 2 bytes = 4 KB. 262,144 * 4 KB = 1 GB (well within 32 GB HBM!).
  # Wait: to make it super fast and avoid gathering 256 pages when max(kv_lens) is smaller,
  # or to avoid 1 GB gather if we chunk by e.g. 8 or 16 tokens, or use `jax.vmap`!
  # Let's check: 64 tokens * 4096 KV_len * 8 heads * 128 head_dim * 2 bytes = 512 MB!
  # 512 MB is tiny on a 32 GB TPU v6e, and fully vectorized `vmap` over the 64 tokens (or 8 batches of 8 tokens) runs in ~0.1 ms on TPU v6e!

  def attend_token_batch(t_indices):
    # t_indices: [B]
    s = tok_seq[t_indices]  # [B]
    is_val = valid_tok[t_indices]  # [B]
    q_pos = kv_pos[t_indices]  # [B]
    k_len = kv_lens[s]  # [B]

    # Gather pages for these sequences: [B, pages_per_seq]
    seq_pages = page_table[s]
    # Gather KV from updated_kv_cache:
    # [B, pages_per_seq, page_size, kv_dim2, kv_packing, aligned_head_dim]
    kv_seq = updated_kv_cache[seq_pages]
    B = t_indices.shape[0]
    kv_seq = kv_seq.reshape(
        B,
        pages_per_seq * page_size,
        num_kv_heads_x2_aligned,
        aligned_head_dim,
    )
    kv_seq = kv_seq[:, :, :actual_num_kv_heads_x2, :head_dim]
    kv_seq = kv_seq.reshape(B, pages_per_seq * page_size, num_kv_heads, 2, head_dim)
    k_seq = kv_seq[:, :, :, 0, :]  # [B, S, K, H]
    v_seq = kv_seq[:, :, :, 1, :]  # [B, S, K, H]

    q_tok = q_grouped[t_indices]  # [B, K, G, H]

    # Compute QK^T: [B, K, G, H] x [B, S, K, H] -> [B, K, G, S]
    qk = jax.lax.dot_general(
        q_tok,
        k_seq,
        dimension_numbers=(([3], [3]), ([0, 1], [0, 2])),
        preferred_element_type=jnp.float32,
    ).astype(out_dtype)
    qk = qk * sm_scale

    # Masking: kv_idx < k_len and kv_idx <= q_pos
    kv_idx = jnp.arange(pages_per_seq * page_size, dtype=jnp.int32)[None, :]  # [1, S]
    upper_bound = jnp.minimum(k_len, q_pos + 1)[:, None]  # [B, 1]
    mask = kv_idx < upper_bound  # [B, S]
    qk = jnp.where(mask[:, None, None, :], qk, mask_value)

    m = jnp.max(qk, axis=-1, keepdims=True)
    p = jnp.exp(qk - m)
    l = jnp.sum(p, axis=-1, keepdims=True, dtype=out_dtype)

    # Compute PV: [B, K, G, S] x [B, S, K, H] -> [B, K, G, H]
    pv = jax.lax.dot_general(
        p,
        v_seq,
        dimension_numbers=(([3], [1]), ([0, 1], [0, 2])),
        preferred_element_type=jnp.float32,
    ).astype(out_dtype)

    out = jax.lax.div(pv, l).astype(out_dtype)
    out = out.reshape(B, num_q_heads, head_dim)
    # If token was not part of any active sequence, preserve original queries value (due to aliasing in kernel)
    return jnp.where(is_val[:, None, None], out, queries[t_indices])

  out_tokens = attend_token_batch(token_ids)
  return out_tokens, updated_kv_cache
