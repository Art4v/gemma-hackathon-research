import jax
import jax.numpy as jnp
import numpy as np
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp


CONFIGS = {
    'decode_small': {
        'name': 'decode_small',
        'model': 'DeepSeek-V4',
        'operator': 'masked_dense_mla',
        'batch_size': 16,
        'q_len': 1,
        'kv_len': 1024,
        'num_q_heads': 128,
        'head_dim': 512,
        'page_size': 256,
        'csa_topk': 256,
        'sm_scale': 1.0,
        'num_kv_pages_per_block': 1,
        'num_queries_per_block': 1,
    },
    'decode_b256_kv9k': {
        'name': 'decode_b256_kv9k',
        'model': 'DeepSeek-V4',
        'operator': 'masked_dense_mla',
        'batch_size': 256,
        'q_len': 1,
        'kv_len': 9216,
        'num_q_heads': 128,
        'head_dim': 512,
        'page_size': 1024,
        'csa_topk': 1024,
        'sm_scale': 1.0,
        'num_kv_pages_per_block': 1,
        'num_queries_per_block': 1,
    },
    'prefill_chunk_first': {
        'name': 'prefill_chunk_first',
        'model': 'DeepSeek-V4',
        'operator': 'masked_dense_mla',
        'batch_size': 1,
        'q_len': 1024,
        'kv_len': 1024,
        'num_q_heads': 128,
        'head_dim': 512,
        'page_size': 1024,
        'csa_topk': 1024,
        'sm_scale': 1.0,
        'num_kv_pages_per_block': 1,
        'num_queries_per_block': 32,
    },
    'prefill_chunk_last': {
        'name': 'prefill_chunk_last',
        'model': 'DeepSeek-V4',
        'operator': 'masked_dense_mla',
        'batch_size': 1,
        'q_len': 1024,
        'kv_len': 8192,
        'num_q_heads': 128,
        'head_dim': 512,
        'page_size': 1024,
        'csa_topk': 1024,
        'sm_scale': 1.0,
        'num_kv_pages_per_block': 2,
        'num_queries_per_block': 32,
    },
}


CONFIG = CONFIGS['decode_small']


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns inputs for Masked-Dense MLA."""
  cfg = (
      CONFIG
      if config is None
      else (CONFIGS[config] if isinstance(config, str) else config)
  )

  key = jax.random.key(42)
  k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = jax.random.split(key, 10)

  B = cfg['batch_size']
  q_len = cfg['q_len']
  num_tokens = B * q_len
  kv_len = cfg['kv_len'] // 4
  num_heads = cfg['num_q_heads']
  head_dim = cfg['head_dim']
  page_size = cfg['page_size'] // 4
  topk = cfg['csa_topk']

  pages_per_seq = (kv_len + page_size - 1) // page_size + 2
  total_pages = B * pages_per_seq

  q = jax.random.normal(k1, (num_tokens, num_heads, head_dim), dtype=dtype)
  cache_kv_nope = jax.random.randint(
      k8, (total_pages, page_size, 4, 128), 1, 126, dtype=jnp.uint8
  )
  rope_bf16 = jax.random.normal(
      k9, (total_pages, page_size, 64), dtype=jnp.bfloat16
  )
  rope_u16 = jax.lax.bitcast_convert_type(rope_bf16, jnp.uint16)
  hi = (rope_u16 >> 8).astype(jnp.uint8)
  lo = (rope_u16 & 0xFF).astype(jnp.uint8)
  cache_kv_rope = jnp.concatenate([hi, lo], axis=-1).reshape(
      total_pages, page_size // 4, 4, 128
  )
  kv_lens = jnp.full((B,), kv_len, dtype=jnp.int32)

  rng = np.random.default_rng(42)
  topk_indices_list = []
  for _ in range(B):
    for j in range(q_len):
      perm = rng.permutation((cfg['kv_len'] - q_len + j) // 4)
      indices = list(perm[:topk])
      if len(indices) < topk:
        indices.extend([-1] * (topk - len(indices)))
      topk_indices_list.append(indices)
  topk_indices = jnp.array(topk_indices_list, dtype=jnp.int32)
  page_indices = jnp.arange(total_pages, dtype=jnp.int32)
  cu_q_lens = jnp.arange(0, num_tokens + 1, q_len, dtype=jnp.int32)
  num_decode_seqs = B if q_len == 1 else 0
  distribution = jnp.array(
      [num_decode_seqs, num_decode_seqs, B], dtype=jnp.int32
  )

  attention_sinks = jax.random.uniform(k5, (num_heads,), dtype=jnp.float32)
  swa_accumulation = jax.random.normal(
      k6, (num_tokens, num_heads, head_dim), dtype=dtype
  )
  swa_l = jax.random.uniform(k7, (num_tokens, num_heads), dtype=jnp.float32)
  swa_m = jax.random.uniform(k10, (num_tokens, num_heads), dtype=jnp.float32)

  return (
      q,
      cache_kv_nope,
      cache_kv_rope,
      kv_lens,
      topk_indices,
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

_DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def _dequant_page(nope_page: jax.Array, rope_page: jax.Array) -> jax.Array:
  """Dequantizes a single KV cache page of shape [page_size, 512] to bfloat16."""
  page_size = nope_page.shape[0]
  nope_fp8 = jax.lax.bitcast_convert_type(
      nope_page[:, :448], jnp.float8_e4m3fn
  ).astype(jnp.bfloat16)
  scale_u16 = (nope_page[:, 448:455].astype(jnp.uint32) << 7).astype(jnp.uint16)
  nope_scales = jax.lax.bitcast_convert_type(scale_u16, jnp.bfloat16)
  nope = (
      nope_fp8.reshape(page_size, 7, 64) * nope_scales[:, :, None]
  ).reshape(page_size, 448)

  hi = rope_page[:, :64].astype(jnp.uint32)
  lo = rope_page[:, 64:128].astype(jnp.uint32)
  rope_u16 = ((hi << 8) | lo).astype(jnp.uint16)
  rope = jax.lax.bitcast_convert_type(rope_u16, jnp.bfloat16)

  return jnp.concatenate([nope, rope], axis=-1)


def workload(
    q: jax.Array,
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
    kv_lens: jax.Array,
    topk_indices: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sinks: jax.Array,
    swa_accumulation: jax.Array,
    swa_l: jax.Array,
    swa_m: jax.Array,
    *,
    num_queries_per_block=None,
    num_kv_pages_per_block=None,
) -> jax.Array:
  del num_queries_per_block, num_kv_pages_per_block
  num_tokens, num_heads, head_dim = q.shape
  total_pages, page_size = cache_kv_nope.shape[0], cache_kv_nope.shape[1]
  max_num_seqs = kv_lens.shape[0]
  pages_per_seq = page_indices.shape[0] // max_num_seqs
  max_kv_len = pages_per_seq * page_size
  sm_scale = 1.0
  mask_value = _DEFAULT_MASK_VALUE

  nope_flat = cache_kv_nope.reshape(total_pages, page_size, -1)
  rope_flat = cache_kv_rope.reshape(total_pages, page_size, 128)
  page_indices_2d = page_indices.reshape(max_num_seqs, pages_per_seq)

  # Map each query token to its sequence index in [0, max_num_seqs).
  token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
  token_seq_idx = (
      jnp.searchsorted(cu_q_lens[1:], token_ids, side="right")
      .astype(jnp.int32)
      .clip(0, max_num_seqs - 1)
  )

  active_end = distribution[2]
  token_active = (
      (token_seq_idx < active_end)
      & (token_ids >= cu_q_lens[token_seq_idx])
      & (token_ids < cu_q_lens[token_seq_idx + 1])
  )

  # Generate binary attention mask [num_tokens, max_kv_len] from topk_indices.
  valid_topk = (topk_indices >= 0) & (topk_indices < max_kv_len)
  safe_topk = jnp.where(valid_topk, topk_indices, 0)
  row_idx = jnp.arange(num_tokens, dtype=jnp.int32)[:, None]
  mask_hbm = jnp.zeros((num_tokens, max_kv_len), dtype=jnp.bool_)
  mask_hbm = mask_hbm.at[row_idx, safe_topk].max(valid_topk)
  mask_pages = mask_hbm.reshape(num_tokens, pages_per_seq, page_size)

  # Determine maximum number of KV pages needed across active sequences.
  seq_Valid = jnp.arange(max_num_seqs, dtype=jnp.int32) < active_end
  active_kv_lens = jnp.where(seq_Valid, kv_lens, 0)
  max_kv_pages = jnp.minimum(
      pages_per_seq,
      jnp.maximum(1, (jnp.max(active_kv_lens) + page_size - 1) // page_size),
  )

  init_m = swa_m.astype(jnp.float32)
  init_l = swa_l.astype(jnp.float32)
  init_acc = swa_accumulation.astype(jnp.float32)

  def page_step(p_idx, carry):
    m_prev, l_prev, acc_prev = carry

    # Gather and dequantize page `p_idx` for all sequences: [B, page_size, head_dim]
    seq_page_ids = jnp.clip(page_indices_2d[:, p_idx], 0, total_pages - 1)
    kv_seq = _dequant_page(
        nope_flat[seq_page_ids].reshape(-1, nope_flat.shape[-1]),
        rope_flat[seq_page_ids].reshape(-1, 128),
    ).reshape(max_num_seqs, page_size, head_dim)

    # Zero out KV positions >= kv_len (prevents NaNs in unused cache slots).
    k_pos = p_idx * page_size + jnp.arange(page_size, dtype=jnp.int32)[None, :]
    kv_valid_seq = k_pos < kv_lens[:, None]
    kv_seq = jnp.where(kv_valid_seq[:, :, None], kv_seq, 0)

    # Gather sequence KV page for each query token: [num_tokens, page_size, head_dim]
    kv_tok = kv_seq[token_seq_idx]

    # Attention scores: [num_tokens, num_heads, page_size]
    s = (
        jnp.einsum(
            "thd,tkd->thk",
            q,
            kv_tok,
            preferred_element_type=jnp.float32,
        )
        * sm_scale
    )

    page_mask = mask_pages[:, p_idx, :]  # [num_tokens, page_size]
    s = jnp.where(page_mask[:, None, :], s, mask_value)

    # Online softmax update (only for pages within the sequence's kv_len).
    page_active = (p_idx == 0) | ((p_idx * page_size) < kv_lens[token_seq_idx])
    s_rowmax = jnp.max(s, axis=-1)
    m_curr = jnp.maximum(m_prev, s_rowmax)
    p = jnp.exp(s - m_curr[:, :, None])
    pv = jnp.einsum(
        "thk,tkd->thd",
        p.astype(jnp.bfloat16),
        kv_tok,
        preferred_element_type=jnp.float32,
    )
    p_rowsum = jnp.sum(p, axis=-1)
    exp_m_diff = jnp.exp(m_prev - m_curr)
    l_curr = exp_m_diff * l_prev + p_rowsum
    acc_curr = exp_m_diff[:, :, None] * acc_prev + pv

    m_next = jnp.where(page_active[:, None], m_curr, m_prev)
    l_next = jnp.where(page_active[:, None], l_curr, l_prev)
    acc_next = jnp.where(page_active[:, None, None], acc_curr, acc_prev)
    return m_next, l_next, acc_next

  m_final, l_final, acc_final = jax.lax.fori_loop(
      0, max_kv_pages, page_step, (init_m, init_l, init_acc)
  )

  exp_sinks = jnp.exp(attention_sinks[None, :] - m_final)
  denom = l_final + exp_sinks
  out = (acc_final / denom[:, :, None]).astype(q.dtype)
  return jnp.where(token_active[:, None, None], out, q)
