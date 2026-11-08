import jax
import jax
import jax.numpy as jnp
import jax.numpy as jnp
import numpy as np
import jax
import jax.numpy as jnp


CONFIGS = {
    'decode_small': {
        'name': 'decode_small',
        'model': 'DeepSeek-V4',
        'operator': 'compressed_sparse_attention',
        'batch_size': 16,
        'q_len': 1,
        'kv_len': 1024,
        'num_q_heads': 128,
        'head_dim': 512,
        'page_size': 256,
        'csa_topk': 256,
        'sm_scale': 1.0,
        'gather_and_attention_chunk_size': 64,
        'attention_kernel_batch_size': 16,
    },
    'dsv4_pro_decode': {
        'name': 'dsv4_pro_decode',
        'model': 'DeepSeek-V4-Pro',
        'operator': 'compressed_sparse_attention',
        'batch_size': 256,
        'q_len': 1,
        'kv_len': 9216,
        'num_q_heads': 128,
        'head_dim': 512,
        'page_size': 1024,
        'csa_topk': 1024,
        'sm_scale': 1.0,
        'gather_and_attention_chunk_size': 64,
        'attention_kernel_batch_size': 16,
    },
    'dsv4_pro_prefill_chunk_first': {
        'name': 'dsv4_pro_prefill_chunk_first',
        'model': 'DeepSeek-V4-Pro',
        'operator': 'compressed_sparse_attention',
        'batch_size': 1,
        'q_len': 1024,
        'kv_len': 1024,
        'num_q_heads': 128,
        'head_dim': 512,
        'page_size': 1024,
        'csa_topk': 1024,
        'sm_scale': 1.0,
        'gather_and_attention_chunk_size': 64,
        'attention_kernel_batch_size': 16,
    },
    'dsv4_pro_prefill_chunk_last': {
        'name': 'dsv4_pro_prefill_chunk_last',
        'model': 'DeepSeek-V4-Pro',
        'operator': 'compressed_sparse_attention',
        'batch_size': 1,
        'q_len': 1024,
        'kv_len': 8192,
        'num_q_heads': 128,
        'head_dim': 512,
        'page_size': 1024,
        'csa_topk': 1024,
        'sm_scale': 1.0,
        'gather_and_attention_chunk_size': 64,
        'attention_kernel_batch_size': 16,
    },
    'dsv4_flash_decode': {
        'name': 'dsv4_flash_decode',
        'model': 'DeepSeek-V4-Flash',
        'operator': 'compressed_sparse_attention',
        'batch_size': 256,
        'q_len': 1,
        'kv_len': 9216,
        'num_q_heads': 64,
        'head_dim': 512,
        'page_size': 1024,
        'csa_topk': 512,
        'sm_scale': 1.0,
        'gather_and_attention_chunk_size': 64,
        'attention_kernel_batch_size': 16,
    },
    'dsv4_flash_prefill_chunk_first': {
        'name': 'dsv4_flash_prefill_chunk_first',
        'model': 'DeepSeek-V4-Flash',
        'operator': 'compressed_sparse_attention',
        'batch_size': 1,
        'q_len': 1024,
        'kv_len': 1024,
        'num_q_heads': 64,
        'head_dim': 512,
        'page_size': 1024,
        'csa_topk': 512,
        'sm_scale': 1.0,
        'gather_and_attention_chunk_size': 64,
        'attention_kernel_batch_size': 16,
    },
    'dsv4_flash_prefill_chunk_last': {
        'name': 'dsv4_flash_prefill_chunk_last',
        'model': 'DeepSeek-V4-Flash',
        'operator': 'compressed_sparse_attention',
        'batch_size': 1,
        'q_len': 1024,
        'kv_len': 8192,
        'num_q_heads': 64,
        'head_dim': 512,
        'page_size': 1024,
        'csa_topk': 512,
        'sm_scale': 1.0,
        'gather_and_attention_chunk_size': 64,
        'attention_kernel_batch_size': 16,
    },
}


CONFIG = CONFIGS['decode_small']


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns inputs tuple for CSA."""
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
  distribution = jnp.array([num_decode_seqs, num_decode_seqs, B], dtype=jnp.int32)

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
      topk_indices,
      page_indices,
      cu_q_lens,
      distribution,
      attention_sinks,
      swa_accumulation,
      swa_l,
      swa_m,
  )


import functools
import jax
from jax import lax
import jax.numpy as jnp


def _dequant_dsv4_fp8(bkv_nope: jax.Array) -> jax.Array:
  """Dequantize DSV4 FP8 nope values (..., 512) uint8 to (..., 448) bfloat16."""
  nope_fp8 = lax.bitcast_convert_type(
      bkv_nope[..., :448], jnp.float8_e4m3fn
  ).astype(jnp.bfloat16)

  scale_u32 = bkv_nope[..., 448:455].astype(jnp.uint32) << 7
  scale_u16 = scale_u32.astype(jnp.uint16)
  nope_scales = lax.bitcast_convert_type(scale_u16, jnp.bfloat16)

  nope_fp8_blocks = nope_fp8.reshape(*nope_fp8.shape[:-1], 7, 64)
  nope = (nope_fp8_blocks * nope_scales[..., None]).reshape(
      *bkv_nope.shape[:-1], 448
  )
  return nope.astype(jnp.bfloat16)


def _unpack_rope(bkv_rope: jax.Array) -> jax.Array:
  """Unpack rope values (..., 128) uint8 to (..., 64) bfloat16."""
  hi = bkv_rope[..., :64].astype(jnp.uint16)
  lo = bkv_rope[..., 64:128].astype(jnp.uint16)
  u16 = (hi << 8) | lo
  return lax.bitcast_convert_type(u16, jnp.bfloat16)


@functools.partial(jax.jit, static_argnames=('sm_scale', 'chunk_size'))
def _sparse_ragged_paged_attention_jax(
    q: jax.Array,
    cache_kv_nope: jax.Array,
    cache_kv_rope: jax.Array,
    topk_indices: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    attention_sinks: jax.Array,
    swa_accumulation: jax.Array,
    swa_l: jax.Array,
    swa_m: jax.Array,
    *,
    sm_scale: float = 1.0,
    chunk_size: int = 64,
) -> jax.Array:
  num_tokens, num_q_heads, head_dim = q.shape
  q_dtype = q.dtype
  total_num_pages, page_size, _, _ = cache_kv_nope.shape
  max_num_seqs = cu_q_lens.shape[0] - 1
  pages_per_seq = page_indices.shape[0] // max_num_seqs
  topk = topk_indices.shape[-1]

  tokens_per_seq = cu_q_lens[1:] - cu_q_lens[:-1]
  seq_ids_segment = jnp.repeat(
      jnp.arange(max_num_seqs, dtype=jnp.int32),
      tokens_per_seq,
      total_repeat_length=num_tokens,
  )

  kv_lens = jnp.sum(topk_indices != -1, axis=-1)
  seq_page_ids = topk_indices // page_size
  token_offset = topk_indices % page_size

  page_table = page_indices.reshape(max_num_seqs, pages_per_seq)
  logical = jnp.clip(seq_page_ids, 0, pages_per_seq - 1)
  page_ids = page_table[seq_ids_segment[:, None], logical]

  is_padding = topk_indices == -1
  flat_element_index = jnp.arange(num_tokens * topk, dtype=jnp.int32).reshape(
      num_tokens, topk
  )
  scattered_page_ids = (flat_element_index * 104729) % total_num_pages
  scattered_token_offset = (flat_element_index * 15485863) % page_size
  page_ids = jnp.where(is_padding, scattered_page_ids, page_ids)
  token_offset = jnp.where(is_padding, scattered_token_offset, token_offset)

  indices = page_ids * page_size + token_offset  # [num_tokens, topk]

  nope_flat = cache_kv_nope.reshape(-1, 512)
  rope_flat = cache_kv_rope.reshape(-1, 128)
  num_valid_tokens = cu_q_lens[distribution[2]]

  def compute_chunk(q_c, idx_c, kv_len_c, swa_acc_c, swa_l_c, swa_m_c):
    nope_g = nope_flat[idx_c]  # [C, topk, 512]
    rope_g = rope_flat[idx_c]  # [C, topk, 128]

    bkv_nope = _dequant_dsv4_fp8(nope_g)  # [C, topk, 448]
    bkv_rope = _unpack_rope(rope_g)  # [C, topk, 64]
    bkv = jnp.concatenate([bkv_nope, bkv_rope], axis=-1)  # [C, topk, 512]

    k_span = jnp.arange(topk, dtype=jnp.int32)[None, :]  # [1, topk]
    valid_mask = k_span < kv_len_c[:, None]  # [C, topk]
    bkv = jnp.where(valid_mask[:, :, None], bkv, 0)

    s = jnp.einsum(
        'cnd,cmd->cnm', q_c, bkv, preferred_element_type=jnp.float32
    )
    s *= sm_scale
    s = jnp.where(valid_mask[:, None, :], s, jnp.finfo(s.dtype).min)

    s_rowmax = jnp.max(s, axis=-1, keepdims=True)  # [C, num_q_heads, 1]
    m_prev = swa_m_c[:, :, None]
    m_curr = jnp.maximum(m_prev, s_rowmax)
    p = jnp.exp(s - m_curr)
    exp_m_diff = jnp.exp(m_prev - m_curr)
    p_rowsum = jnp.sum(p, axis=-1, keepdims=True)
    l_prev = swa_l_c[:, :, None]
    l_curr = exp_m_diff * l_prev + p_rowsum
    exp_attention_sinks = jnp.exp(attention_sinks[None, :, None] - m_curr)
    l = l_curr + exp_attention_sinks

    pv = jnp.einsum(
        'cnm,cmd->cnd', p, bkv, preferred_element_type=jnp.float32
    )
    o_prev = swa_acc_c.astype(jnp.float32)
    acc = exp_m_diff * o_prev + pv
    out = (acc / l).astype(q_dtype)
    return out

  if num_tokens <= chunk_size:
    out_all = compute_chunk(
        q, indices, kv_lens, swa_accumulation, swa_l, swa_m
    )
  else:
    num_chunks = (num_tokens + chunk_size - 1) // chunk_size
    padded_tokens = num_chunks * chunk_size
    pad_len = padded_tokens - num_tokens

    def pad_arr(x):
      if pad_len == 0:
        return x
      return jnp.pad(x, ((0, pad_len),) + ((0, 0),) * (x.ndim - 1))

    q_p = pad_arr(q).reshape(num_chunks, chunk_size, num_q_heads, head_dim)
    idx_p = pad_arr(indices).reshape(num_chunks, chunk_size, topk)
    kv_lens_p = pad_arr(kv_lens).reshape(num_chunks, chunk_size)
    swa_acc_p = pad_arr(swa_accumulation).reshape(
        num_chunks, chunk_size, num_q_heads, head_dim
    )
    swa_l_p = pad_arr(swa_l).reshape(num_chunks, chunk_size, num_q_heads)
    swa_m_p = pad_arr(swa_m).reshape(num_chunks, chunk_size, num_q_heads)

    def scan_body(carry, xs):
      q_c, idx_c, kv_len_c, swa_acc_c, swa_l_c, swa_m_c = xs
      out_c = compute_chunk(
          q_c, idx_c, kv_len_c, swa_acc_c, swa_l_c, swa_m_c
      )
      return carry, out_c

    _, out_p = lax.scan(
        scan_body,
        None,
        (q_p, idx_p, kv_lens_p, swa_acc_p, swa_l_p, swa_m_p),
    )
    out_all = out_p.reshape(padded_tokens, num_q_heads, head_dim)[:num_tokens]

  token_idx = jnp.arange(num_tokens, dtype=jnp.int32)[:, None, None]
  return jnp.where(token_idx < num_valid_tokens, out_all, q)


def workload(
    q,
    cache_kv_nope,
    cache_kv_rope,
    topk_indices,
    page_indices,
    cu_q_lens,
    distribution,
    attention_sinks,
    swa_accumulation,
    swa_l,
    swa_m,
):
  """Plain JAX workload execution."""
  return _sparse_ragged_paged_attention_jax(
      q,
      cache_kv_nope,
      cache_kv_rope,
      topk_indices,
      page_indices,
      cu_q_lens,
      distribution,
      attention_sinks,
      swa_accumulation,
      swa_l,
      swa_m,
      sm_scale=1.0,
      chunk_size=64,
  )
