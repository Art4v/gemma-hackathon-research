import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import jax.numpy as jnp
import jax
import jax.numpy as jnp


def cdiv(a, b):
  assert b != 0
  return (a + b - 1) // b


def align_to(x, a):
  return cdiv(x, a) * a


def get_dtype_bitwidth(dtype):
  return jax.numpy.dtype(dtype).itemsize * 8


def get_dtype_packing(dtype):
  bits = get_dtype_bitwidth(dtype)
  return 32 // bits


CONFIGS = {
    'llama3_70b_mixed': {
        'name': 'rpa_v3_llama3_70b_mixed',
        'model': 'Llama-3.1-70B',
        'operator': 'ragged_paged_attention_v3',
        'max_num_batched_tokens': 4096,
        'max_num_seqs': 64,
        'num_q_heads': 64,
        'num_kv_heads': 8,
        'head_dim': 128,
        'page_size': 16,
        'pages_per_seq': 256,
        'chunk_prefill_size': 512,
        'active_q_lens': (1,) * 48 + (512,) * 7 + (464,),
        'active_kv_lens': (
            tuple(257 + ((i * 73) % 240) * 16 for i in range(48))
            + (1023, 1535, 2047, 2559, 3071, 3583, 4095, 4095)
        ),
        'distribution': (48, 55, 56),
        'atol': 0.2,
        'rtol': 0.2,
    },
    'llama3_8b_decode': {
        'name': 'rpa_v3_llama3_8b_decode',
        'model': 'Llama-3.1-8B',
        'operator': 'ragged_paged_attention_v3',
        'max_num_batched_tokens': 64,
        'max_num_seqs': 64,
        'num_q_heads': 32,
        'num_kv_heads': 8,
        'head_dim': 128,
        'page_size': 16,
        'pages_per_seq': 256,
        'chunk_prefill_size': None,
        'active_q_lens': (1,) * 64,
        'active_kv_lens': tuple(
            128 + ((i * 37) % 120) * 16 for i in range(64)
        ),
        'distribution': (64, 64, 64),
        'atol': 0.2,
        'rtol': 0.2,
    },
    'gemma2_9b_hd64': {
        'name': 'rpa_v3_gemma2_9b_hd64',
        'model': 'Gemma-2-9B',
        'operator': 'ragged_paged_attention_v3',
        'max_num_batched_tokens': 1024,
        'max_num_seqs': 32,
        'num_q_heads': 16,
        'num_kv_heads': 8,
        'head_dim': 64,
        'page_size': 16,
        'pages_per_seq': 128,
        'chunk_prefill_size': None,
        'active_q_lens': (1,) * 16 + (256,) * 3 + (240,),
        'active_kv_lens': (
            tuple(128 + ((i * 37) % 60) * 16 for i in range(16))
            + (512, 768, 1024, 1024)
        ),
        'distribution': (16, 19, 20),
        'atol': 0.2,
        'rtol': 0.2,
    },
}


CONFIG = CONFIGS['llama3_70b_mixed']


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Create inputs matching the TPU kernel specifications for RPAv3."""
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

  if d == 64:
    padded_num_kv = align_to(h_kv, kv_packing)
    cache_shape = (
        total_pages,
        page_size,
        padded_num_kv // kv_packing,
        kv_packing,
        128,
    )
  else:
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


def _cdiv(a, b):
  return (a + b - 1) // b


def _align_to(x, a):
  return _cdiv(x, a) * a


def _get_dtype_packing(dtype):
  return 32 // (jnp.dtype(dtype).itemsize * 8)


def _merge_kv_hd64(k: jax.Array, v: jax.Array):
  max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
  kv_packing = _get_dtype_packing(k.dtype)
  num_kv_heads = _align_to(actual_num_kv_heads, kv_packing)
  kv = jnp.pad(
      jnp.concat([k, v], axis=-1),
      (
          (0, 0),
          (0, num_kv_heads - actual_num_kv_heads),
          (0, 0),
      ),
      constant_values=0,
  ).reshape(
      max_num_tokens,
      num_kv_heads // kv_packing,
      kv_packing,
      actual_head_dim * 2,
  )
  return kv


def _merge_kv_general(k: jax.Array, v: jax.Array):
  max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
  kv_packing = _get_dtype_packing(k.dtype)
  actual_num_kv_heads_x2 = actual_num_kv_heads * 2
  num_kv_heads_x2 = _align_to(actual_num_kv_heads_x2, kv_packing)
  head_dim = _align_to(actual_head_dim, 128)
  kv = jnp.pad(
      jnp.concat([k, v], axis=-1).reshape(
          max_num_tokens, actual_num_kv_heads_x2, actual_head_dim
      ),
      (
          (0, 0),
          (0, num_kv_heads_x2 - actual_num_kv_heads_x2),
          (0, head_dim - actual_head_dim),
      ),
      constant_values=0,
  ).reshape(
      max_num_tokens,
      num_kv_heads_x2 // kv_packing,
      kv_packing,
      head_dim,
  )
  return kv


def _update_kv_cache_vectorized(
    merged_kv, kv_cache, kv_lens, page_indices, cu_q_lens, num_seqs
):
  max_tokens = merged_kv.shape[0]
  total_pages, page_size = kv_cache.shape[:2]
  max_seqs = kv_lens.shape[0]
  pages_per_seq = page_indices.shape[0] // max_seqs

  token_idx = jnp.arange(max_tokens, dtype=jnp.int32)
  seq_ids = jnp.searchsorted(cu_q_lens, token_idx, side='right') - 1
  seq_ids_clamped = jnp.clip(seq_ids, 0, max_seqs - 1)

  valid_token = (seq_ids >= 0) & (seq_ids < num_seqs) & (token_idx < cu_q_lens[num_seqs])

  q_start = cu_q_lens[seq_ids_clamped]
  q_end = cu_q_lens[seq_ids_clamped + 1]
  q_len = q_end - q_start
  kv_len = kv_lens[seq_ids_clamped]

  pos_in_seq = (kv_len - q_len) + (token_idx - q_start)
  page_in_seq = pos_in_seq // page_size
  offset_in_page = pos_in_seq % page_size

  page_table_idx = jnp.clip(
      seq_ids_clamped * pages_per_seq + page_in_seq,
      0,
      page_indices.shape[0] - 1,
  )
  phys_page = page_indices[page_table_idx]

  flat_cache_pos = phys_page * page_size + offset_in_page
  flat_cache_pos = jnp.clip(flat_cache_pos, 0, total_pages * page_size - 1)

  cache_flat = kv_cache.reshape(total_pages * page_size, *kv_cache.shape[2:])
  safe_indices = jnp.where(valid_token, flat_cache_pos, total_pages * page_size)
  cache_flat = cache_flat.at[safe_indices].set(merged_kv, mode='drop')

  return cache_flat.reshape(kv_cache.shape)


def _rpa_general(
    queries,
    keys,
    values,
    kv_cache,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    sm_scale,
):
  max_tokens, actual_num_q_heads, actual_head_dim = queries.shape
  actual_num_kv_heads = keys.shape[1]
  actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

  total_pages, page_size, num_kv_heads_x2_per_packing, kv_packing, head_dim = (
      kv_cache.shape
  )
  num_kv_heads_x2 = num_kv_heads_x2_per_packing * kv_packing
  max_seqs = kv_lens.shape[0]
  pages_per_seq = page_indices.shape[0] // max_seqs
  max_kv_len = pages_per_seq * page_size

  num_seqs = distribution[2]
  decode_end = distribution[0]

  merged_kv = _merge_kv_general(keys, values)
  kv_cache = _update_kv_cache_vectorized(
      merged_kv, kv_cache, kv_lens, page_indices, cu_q_lens, num_seqs
  )

  out_dtype = jnp.float32 if queries.dtype == jnp.float32 else jnp.bfloat16
  mask_value = float(jnp.finfo(out_dtype).min)

  bq_sz = min(1024 // actual_num_q_heads_per_kv_head, max(1, max_tokens))
  bkv_sz = min(_align_to(1024, page_size), max_kv_len)
  bkv_p = bkv_sz // page_size

  token_idx = jnp.arange(max_tokens, dtype=jnp.int32)
  token_seq_ids = jnp.searchsorted(cu_q_lens, token_idx, side='right') - 1
  token_seq_ids_clamped = jnp.clip(token_seq_ids, 0, max_seqs - 1)
  valid_token = (
      (token_seq_ids >= 0)
      & (token_seq_ids < num_seqs)
      & (token_idx < cu_q_lens[num_seqs])
  )
  is_decode_token = valid_token & (token_seq_ids < decode_end)

  def compute_decode_all():
    num_decode = jnp.minimum(decode_end, max_seqs)

    def decode_one_seq(seq_idx):
      q_start = cu_q_lens[seq_idx]
      kv_len = kv_lens[seq_idx]
      q_seq = queries[q_start].reshape(
          actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim
      )

      seq_pages = jax.lax.dynamic_slice_in_dim(
          page_indices, seq_idx * pages_per_seq, pages_per_seq, axis=0
      )
      gathered_kv = kv_cache[seq_pages]
      kv = gathered_kv.reshape(max_kv_len, num_kv_heads_x2, head_dim)[
          :, : actual_num_kv_heads * 2, :
      ].reshape(max_kv_len, actual_num_kv_heads, 2, head_dim)
      k = kv[:, :, 0, :actual_head_dim]
      v = kv[:, :, 1, :actual_head_dim]

      s = jnp.einsum(
          'hqd,khd->hqk', q_seq, k, preferred_element_type=jnp.float32
      ).astype(out_dtype)
      s = s * sm_scale

      k_pos = jnp.arange(max_kv_len, dtype=jnp.int32)[None, None, :]
      mask = k_pos < kv_len
      s = jnp.where(mask, s, mask_value)

      p = jax.nn.softmax(s.astype(jnp.float32), axis=-1).astype(v.dtype)
      v_masked = jnp.where(
          (jnp.arange(max_kv_len, dtype=jnp.int32) < kv_len)[:, None, None],
          v,
          0,
      )
      out = jnp.einsum(
          'hqk,khd->hqd', p, v_masked, preferred_element_type=jnp.float32
      ).astype(queries.dtype)
      return out.reshape(actual_num_q_heads, actual_head_dim)

    seq_indices = jnp.arange(max_seqs, dtype=jnp.int32)
    safe_seq_indices = jnp.where(seq_indices < num_decode, seq_indices, 0)
    decode_outs = jax.vmap(decode_one_seq)(safe_seq_indices)
    return decode_outs[token_seq_ids_clamped]

  out_decode = jax.lax.cond(
      decode_end > 0,
      compute_decode_all,
      lambda: jnp.zeros_like(queries),
  )
  output = jnp.where(is_decode_token[:, None, None], out_decode, 0)

  def seq_loop_body(seq_idx, out_buf):
    q_start = cu_q_lens[seq_idx]
    q_end = cu_q_lens[seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[seq_idx]
    kv_q_gap = kv_len - q_len
    num_bq = _cdiv(q_len, bq_sz)

    def bq_loop_body(bq_idx, out_buf_inner):
      q_Block_start = q_start + bq_idx * bq_sz
      safe_q_start = jnp.minimum(q_Block_start, max_tokens - bq_sz)
      bq = jax.lax.dynamic_slice_in_dim(queries, safe_q_start, bq_sz, axis=0)
      q_offset = q_Block_start - safe_q_start

      q_token_pos_in_slice = jnp.arange(bq_sz, dtype=jnp.int32) - q_offset
      q_pos = kv_q_gap + bq_idx * bq_sz + q_token_pos_in_slice
      q_valid = (q_token_pos_in_slice >= 0) & (
          bq_idx * bq_sz + q_token_pos_in_slice < q_len
      )

      bq_grouped = bq.reshape(
          bq_sz,
          actual_num_kv_heads,
          actual_num_q_heads_per_kv_head,
          actual_head_dim,
      ).transpose(1, 0, 2, 3)

      m_init = jnp.full(
          (actual_num_kv_heads, bq_sz, actual_num_q_heads_per_kv_head, 1),
          -jnp.inf,
          dtype=jnp.float32,
      )
      l_init = jnp.zeros(
          (actual_num_kv_heads, bq_sz, actual_num_q_heads_per_kv_head, 1),
          dtype=jnp.float32,
      )
      acc_init = jnp.zeros(
          (
              actual_num_kv_heads,
              bq_sz,
              actual_num_q_heads_per_kv_head,
              actual_head_dim,
          ),
          dtype=jnp.float32,
      )

      effective_kv_len = jnp.minimum(kv_len, kv_q_gap + (bq_idx + 1) * bq_sz)
      end_bkv_idx = _cdiv(effective_kv_len, bkv_sz)

      def bkv_loop_body(bkv_idx, state):
        m_prev, l_prev, acc_prev = state
        page_start = seq_idx * pages_per_seq + bkv_idx * bkv_p
        safe_page_start = jnp.minimum(page_start, page_indices.shape[0] - bkv_p)
        block_pages = jax.lax.dynamic_slice_in_dim(
            page_indices, safe_page_start, bkv_p, axis=0
        )

        gathered_kv = kv_cache[block_pages]
        kv = gathered_kv.reshape(bkv_sz, num_kv_heads_x2, head_dim)[
            :, : actual_num_kv_heads * 2, :
        ].reshape(bkv_sz, actual_num_kv_heads, 2, head_dim)
        k = kv[:, :, 0, :actual_head_dim]
        v = kv[:, :, 1, :actual_head_dim]

        s = jnp.einsum(
            'hqgd,khd->hqgk',
            bq_grouped,
            k,
            preferred_element_type=jnp.float32,
        ).astype(out_dtype)
        s = (s * sm_scale).astype(jnp.float32)

        k_pos = bkv_idx * bkv_sz + jnp.arange(bkv_sz, dtype=jnp.int32)
        mask = (q_pos[None, :, None, None] >= k_pos[None, None, None, :]) & (
            k_pos[None, None, None, :] < effective_kv_len
        )
        s = jnp.where(mask, s, mask_value)

        s_max = jnp.max(s, axis=-1, keepdims=True)
        m_curr = jnp.maximum(m_prev, s_max)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        p = jnp.exp(s - m_curr)
        l_curr = exp_m_diff * l_prev + jnp.sum(p, axis=-1, keepdims=True)

        v_valid = (k_pos < effective_kv_len)[:, None, None]
        v_masked = jnp.where(v_valid, v, 0)
        pv = jnp.einsum(
            'hqgk,khd->hqgd',
            p.astype(v.dtype),
            v_masked,
            preferred_element_type=jnp.float32,
        )
        acc_curr = exp_m_diff * acc_prev + pv
        return (m_curr, l_curr, acc_curr)

      _, l_final, acc_final = jax.lax.fori_loop(
          0, end_bkv_idx, bkv_loop_body, (m_init, l_init, acc_init)
      )
      out_block = (acc_final / l_final).astype(queries.dtype)
      out_block = out_block.transpose(1, 0, 2, 3).reshape(
          bq_sz, actual_num_q_heads, actual_head_dim
      )

      existing = jax.lax.dynamic_slice_in_dim(
          out_buf_inner, safe_q_start, bq_sz, axis=0
      )
      merged = jnp.where(q_valid[:, None, None], out_block, existing)
      return jax.lax.dynamic_update_slice_in_dim(
          out_buf_inner, merged, safe_q_start, axis=0
      )

    return jax.lax.fori_loop(0, num_bq, bq_loop_body, out_buf)

  output = jax.lax.fori_loop(decode_end, num_seqs, seq_loop_body, output)
  return output, kv_cache


def _rpa_hd64(
    queries,
    keys,
    values,
    kv_cache,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    sm_scale,
):
  max_tokens, actual_num_q_heads, actual_head_dim = queries.shape
  actual_num_kv_heads = keys.shape[1]
  actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads

  total_pages, page_size, num_kv_heads_per_packing, kv_packing, _ = (
      kv_cache.shape
  )
  padded_num_kv_heads = num_kv_heads_per_packing * kv_packing
  max_seqs = kv_lens.shape[0]
  pages_per_seq = page_indices.shape[0] // max_seqs
  max_kv_len = pages_per_seq * page_size

  num_seqs = distribution[2]
  decode_end = distribution[0]

  merged_kv = _merge_kv_hd64(keys, values)
  kv_cache = _update_kv_cache_vectorized(
      merged_kv, kv_cache, kv_lens, page_indices, cu_q_lens, num_seqs
  )

  mask_value = -0.7 * float(jnp.finfo(jnp.dtype('float32')).max)

  bq_sz = min(64, max(1, max_tokens))
  bkv_sz = min(_align_to(1024, page_size), max_kv_len)
  bkv_p = bkv_sz // page_size

  token_idx = jnp.arange(max_tokens, dtype=jnp.int32)
  token_seq_ids = jnp.searchsorted(cu_q_lens, token_idx, side='right') - 1
  token_seq_ids_clamped = jnp.clip(token_seq_ids, 0, max_seqs - 1)
  valid_token = (
      (token_seq_ids >= 0)
      & (token_seq_ids < num_seqs)
      & (token_idx < cu_q_lens[num_seqs])
  )
  is_decode_token = valid_token & (token_seq_ids < decode_end)

  def compute_decode_all():
    num_decode = jnp.minimum(decode_end, max_seqs)

    def decode_one_seq(seq_idx):
      q_start = cu_q_lens[seq_idx]
      kv_len = kv_lens[seq_idx]
      q_seq = queries[q_start].reshape(
          actual_num_kv_heads, actual_num_q_heads_per_kv_head, actual_head_dim
      )

      seq_pages = jax.lax.dynamic_slice_in_dim(
          page_indices, seq_idx * pages_per_seq, pages_per_seq, axis=0
      )
      gathered_kv = kv_cache[seq_pages]
      kv = gathered_kv.reshape(max_kv_len, padded_num_kv_heads, 128)[
          :, :actual_num_kv_heads, :
      ]
      k = kv[:, :, :64]
      v = kv[:, :, 64:]

      s = jnp.einsum(
          'hqd,khd->hqk', q_seq, k, preferred_element_type=jnp.float32
      )
      s = s * sm_scale

      k_pos = jnp.arange(max_kv_len, dtype=jnp.int32)[None, None, :]
      mask = k_pos < kv_len
      s = jnp.where(mask, s, mask_value)

      p = jax.nn.softmax(s, axis=-1).astype(v.dtype)
      out = jnp.einsum(
          'hqk,khd->hqd', p, v, preferred_element_type=jnp.float32
      ).astype(queries.dtype)
      return out.reshape(actual_num_q_heads, actual_head_dim)

    seq_indices = jnp.arange(max_seqs, dtype=jnp.int32)
    safe_seq_indices = jnp.where(seq_indices < num_decode, seq_indices, 0)
    decode_outs = jax.vmap(decode_one_seq)(safe_seq_indices)
    return decode_outs[token_seq_ids_clamped]

  out_decode = jax.lax.cond(
      decode_end > 0,
      compute_decode_all,
      lambda: jnp.zeros_like(queries),
  )
  output = jnp.where(is_decode_token[:, None, None], out_decode, 0)

  def seq_loop_body(seq_idx, out_buf):
    q_start = cu_q_lens[seq_idx]
    q_end = cu_q_lens[seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[seq_idx]
    kv_q_gap = kv_len - q_len
    num_bq = _cdiv(q_len, bq_sz)
    num_bkv = _cdiv(kv_len, bkv_sz)

    def bq_loop_body(bq_idx, out_buf_inner):
      q_Block_start = q_start + bq_idx * bq_sz
      safe_q_start = jnp.minimum(q_Block_start, max_tokens - bq_sz)
      bq = jax.lax.dynamic_slice_in_dim(queries, safe_q_start, bq_sz, axis=0)
      q_offset = q_Block_start - safe_q_start

      q_token_pos_in_slice = jnp.arange(bq_sz, dtype=jnp.int32) - q_offset
      q_pos = kv_q_gap + bq_idx * bq_sz + q_token_pos_in_slice
      q_valid = (q_token_pos_in_slice >= 0) & (
          bq_idx * bq_sz + q_token_pos_in_slice < q_len
      )

      bq_grouped = bq.reshape(
          bq_sz,
          actual_num_kv_heads,
          actual_num_q_heads_per_kv_head,
          actual_head_dim,
      ).transpose(1, 0, 2, 3)

      m_init = jnp.full(
          (actual_num_kv_heads, bq_sz, actual_num_q_heads_per_kv_head, 1),
          -jnp.inf,
          dtype=jnp.float32,
      )
      l_init = jnp.zeros(
          (actual_num_kv_heads, bq_sz, actual_num_q_heads_per_kv_head, 1),
          dtype=jnp.float32,
      )
      acc_init = jnp.zeros(
          (
              actual_num_kv_heads,
              bq_sz,
              actual_num_q_heads_per_kv_head,
              actual_head_dim,
          ),
          dtype=jnp.float32,
      )

      def bkv_loop_body(bkv_idx, state):
        m_prev, l_prev, acc_prev = state
        page_start = seq_idx * pages_per_seq + bkv_idx * bkv_p
        safe_page_start = jnp.minimum(page_start, page_indices.shape[0] - bkv_p)
        block_pages = jax.lax.dynamic_slice_in_dim(
            page_indices, safe_page_start, bkv_p, axis=0
        )

        gathered_kv = kv_cache[block_pages]
        kv = gathered_kv.reshape(bkv_sz, padded_num_kv_heads, 128)[
            :, :actual_num_kv_heads, :
        ]
        k = kv[:, :, :64]
        v = kv[:, :, 64:]

        s = jnp.einsum(
            'hqgd,khd->hqgk',
            bq_grouped,
            k,
            preferred_element_type=jnp.float32,
        )
        s = s * sm_scale

        k_pos = bkv_idx * bkv_sz + jnp.arange(bkv_sz, dtype=jnp.int32)
        mask = (k_pos[None, None, None, :] <= q_pos[None, :, None, None]) & (
            k_pos[None, None, None, :] < kv_len
        )
        s = jnp.where(mask, s, mask_value)

        s_max = jnp.max(s, axis=-1, keepdims=True)
        m_curr = jnp.maximum(m_prev, s_max)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        p = jnp.exp(s - m_curr)
        l_curr = exp_m_diff * l_prev + jnp.sum(p, axis=-1, keepdims=True)

        v_valid = (k_pos < kv_len)[:, None, None]
        v_masked = jnp.where(v_valid, v, 0)
        pv = jnp.einsum(
            'hqgk,khd->hqgd',
            p.astype(v.dtype),
            v_masked,
            preferred_element_type=jnp.float32,
        )
        acc_curr = exp_m_diff * acc_prev + pv
        return (m_curr, l_curr, acc_curr)

      _, l_final, acc_final = jax.lax.fori_loop(
          0, num_bkv, bkv_loop_body, (m_init, l_init, acc_init)
      )
      out_block = (acc_final / l_final).astype(queries.dtype)
      out_block = out_block.transpose(1, 0, 2, 3).reshape(
          bq_sz, actual_num_q_heads, actual_head_dim
      )

      existing = jax.lax.dynamic_slice_in_dim(
          out_buf_inner, safe_q_start, bq_sz, axis=0
      )
      merged = jnp.where(q_valid[:, None, None], out_block, existing)
      return jax.lax.dynamic_update_slice_in_dim(
          out_buf_inner, merged, safe_q_start, axis=0
      )

    return jax.lax.fori_loop(0, num_bq, bq_loop_body, out_buf)

  output = jax.lax.fori_loop(decode_end, num_seqs, seq_loop_body, output)
  return output, kv_cache


def workload(
    queries,
    keys,
    values,
    kv_cache,
    kv_lens,
    page_indices,
    cu_q_lens,
    distribution,
    chunk_prefill_size=None,
):
  """Plain JAX implementation of ragged paged attention v3."""
  del chunk_prefill_size
  actual_head_dim = queries.shape[2]
  sm_scale = 1.0 / math.sqrt(actual_head_dim)
  if actual_head_dim == 64:
    return _rpa_hd64(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale,
    )
  else:
    return _rpa_general(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale,
    )
