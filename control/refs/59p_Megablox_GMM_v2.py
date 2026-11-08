import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp


def sub_channel_quantize(x: jax.Array, quant_dtype: jnp.dtype, wsz: int = 128):
  """Quantizes x with sub-channel quantization on the 2nd minor."""
  if jnp.issubdtype(quant_dtype, jnp.floating):
    dtype_info = jnp.finfo(quant_dtype)
  else:
    dtype_info = jnp.iinfo(quant_dtype)
  dtype_max = float(dtype_info.max)
  w_lst, scale_lst = [], []
  assert len(x.shape) >= 2
  assert x.shape[-2] % wsz == 0
  for i in range(0, x.shape[1], wsz):
    y = x[:, i : i + wsz, :]
    abs_max = jnp.abs(y).max(axis=1, keepdims=True)
    scale = (abs_max / dtype_max).astype(jnp.float32)
    w = (y / scale).astype(quant_dtype)
    w_lst.append(w)
    scale_lst.append(scale)
  return jnp.concat(w_lst, axis=1), jnp.concat(scale_lst, axis=1)


def _parse_dtype(dtype_str: str) -> jnp.dtype:
  if dtype_str == 'bf16':
    return jnp.bfloat16
  elif dtype_str == 'fp8':
    return jnp.float8_e4m3fn
  elif dtype_str == 'fp4':
    return jnp.float4_e2m1fn
  elif dtype_str == 'int4':
    return jnp.int4
  else:
    raise ValueError(f'Unsupported dtype: {dtype_str}')


CONFIGS = {
    'gmm_small': {
        'name': 'gmm_small',
        'model': 'Megablox-GMM',
        'operator': 'gmm_v2',
        'num_tokens': 2048,
        'hidden_size': 2048,
        'intermediate_size': 1024,
        'global_num_experts': 64,
        'local_num_experts': 8,
        'top_k': 4,
        'weight_dtype': 'fp4',
        'activation_dtype': 'fp8',
        'quant_block_size': -1,
        'atol': 0.1,
        'rtol': 0.1,
    },
    'dsv4_pro_16k': {
        'name': 'dsv4_pro_16k',
        'model': 'DeepSeek-V4-Pro',
        'operator': 'gmm_v2',
        'num_tokens': 16384,
        'hidden_size': 7168,
        'intermediate_size': 4096,
        'global_num_experts': 256,
        'local_num_experts': 16,
        'top_k': 8,
        'weight_dtype': 'fp4',
        'activation_dtype': 'fp8',
        'quant_block_size': -1,
        'atol': 0.1,
        'rtol': 0.1,
    },
    'dsv4_pro_8k': {
        'name': 'dsv4_pro_8k',
        'model': 'DeepSeek-V4-Pro',
        'operator': 'gmm_v2',
        'num_tokens': 8192,
        'hidden_size': 7168,
        'intermediate_size': 4096,
        'global_num_experts': 256,
        'local_num_experts': 16,
        'top_k': 8,
        'weight_dtype': 'fp4',
        'activation_dtype': 'fp8',
        'quant_block_size': -1,
        'atol': 0.1,
        'rtol': 0.1,
    },
    'dsv4_flash_16k': {
        'name': 'dsv4_flash_16k',
        'model': 'DeepSeek-V4-Flash',
        'operator': 'gmm_v2',
        'num_tokens': 16384,
        'hidden_size': 4096,
        'intermediate_size': 2048,
        'global_num_experts': 256,
        'local_num_experts': 16,
        'top_k': 8,
        'weight_dtype': 'fp4',
        'activation_dtype': 'fp8',
        'quant_block_size': -1,
        'atol': 0.1,
        'rtol': 0.1,
    },
    'dsv4_flash_8k': {
        'name': 'dsv4_flash_8k',
        'model': 'DeepSeek-V4-Flash',
        'operator': 'gmm_v2',
        'num_tokens': 8192,
        'hidden_size': 4096,
        'intermediate_size': 2048,
        'global_num_experts': 256,
        'local_num_experts': 16,
        'top_k': 8,
        'weight_dtype': 'fp4',
        'activation_dtype': 'fp8',
        'quant_block_size': -1,
        'atol': 0.1,
        'rtol': 0.1,
    },
}


CONFIG = CONFIGS['gmm_small']


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns (lhs, rhs, group_sizes, rhs_scale, rhs_bias, group_offset)."""
  cfg = (
      CONFIG
      if config is None
      else (CONFIGS[config] if isinstance(config, str) else config)
  )

  key = jax.random.key(42)
  k1, k2, k3 = jax.random.split(key, 3)
  M = cfg['num_tokens']
  K = cfg['hidden_size']
  N = cfg['intermediate_size']
  G_local = cfg['local_num_experts']
  G_global = cfg['global_num_experts']
  weight_dtype = _parse_dtype(cfg.get('weight_dtype', 'fp4'))
  quant_block_size = cfg.get('quant_block_size', -1)

  lhs = (jax.random.uniform(k1, (M, K), dtype=dtype) - 0.5) * 0.2
  rhs = (jax.random.normal(k2, (G_local, K, N), dtype=dtype)) * 0.02

  tokens_per_expert = M // G_global
  remainder = M % G_global
  sizes = [
      tokens_per_expert + (1 if i < remainder else 0) for i in range(G_global)
  ]
  group_sizes = jnp.array(sizes, dtype=jnp.int32)
  group_offset = jnp.array([0], dtype=jnp.int32)

  rhs_scale = None
  if weight_dtype != jnp.bfloat16:
    rhs_wsz = quant_block_size if quant_block_size > 0 else rhs.shape[1]
    rhs, rhs_scale = sub_channel_quantize(rhs, weight_dtype, rhs_wsz)
    rhs_scale = jnp.expand_dims(rhs_scale, axis=2)

  rhs_bias = None
  return lhs, rhs, group_sizes, rhs_scale, rhs_bias, group_offset


import jax
import jax.numpy as jnp


def workload(
    lhs, rhs, group_sizes, rhs_scale=None, rhs_bias=None, group_offset=None
):
  """Plain JAX implementation of Megablox GMM v2."""
  size_m, size_k = lhs.shape
  size_group, _, size_n = rhs.shape
  size_lhs_group = group_sizes.shape[0]
  out_dtype = lhs.dtype

  if group_offset is None:
    go = jnp.int32(0)
  elif jnp.ndim(group_offset) == 0:
    go = group_offset.astype(jnp.int32)
  else:
    go = group_offset[0].astype(jnp.int32)

  # Determine quantization and dequantization configurations (matching TPU v6e)
  mxu_size = 128
  has_scale = rhs_scale is not None
  if has_scale:
    rhs_scale_f32 = rhs_scale.astype(jnp.float32)
    num_blocks = rhs_scale.shape[1]
    rhs_qbs = size_k // num_blocks
    dequant_before = rhs_qbs < mxu_size
    dequant_after = not dequant_before
  else:
    rhs_scale_f32 = None
    num_blocks = 1
    rhs_qbs = size_k
    dequant_before = False
    dequant_after = False

  lhs_q_dtype = None
  if dequant_after:
    is_rhs_float = jnp.issubdtype(rhs.dtype, jnp.floating)
    is_rhs_4bits = jax.dtypes.itemsize_bits(rhs.dtype) == 4
    if is_rhs_float or is_rhs_4bits:
      lhs_q_dtype = jnp.float8_e4m3fn
    else:
      lhs_q_dtype = jnp.int8

  should_quantize = (lhs_q_dtype is not None) and (lhs_q_dtype != lhs.dtype)
  acc_dtype = jnp.bfloat16 if lhs_q_dtype is not None else jnp.float32

  # Step 1: Dequantize RHS before matmul if block size < mxu_size
  if dequant_before:
    rhs_proc = rhs.astype(lhs.dtype).reshape(
        size_group, num_blocks, rhs_qbs, size_n
    )
    rhs_proc = (rhs_proc * rhs_scale_f32.astype(lhs.dtype)).reshape(
        size_group, size_k, size_n
    )
  else:
    rhs_proc = rhs

  # Step 2: Quantize LHS if required
  if should_quantize:
    q_block_size = 512
    pad_k = (-size_k) % q_block_size
    if pad_k > 0:
      lhs_padded = jnp.pad(lhs, ((0, 0), (0, pad_k)))
    else:
      lhs_padded = lhs
    num_l_blocks = lhs_padded.shape[1] // q_block_size
    lhs_blocks = lhs_padded.reshape(size_m, num_l_blocks, q_block_size)

    if jnp.issubdtype(lhs_q_dtype, jnp.floating):
      dtype_max = float(jnp.finfo(lhs_q_dtype).max)
    else:
      dtype_max = float(jnp.iinfo(lhs_q_dtype).max)

    abs_max = jnp.max(jnp.abs(lhs_blocks), axis=2, keepdims=True)
    l_scale = abs_max / dtype_max
    l_scale_inv = jnp.where(l_scale == 0, 0, 1.0 / l_scale)
    lhs_q = (lhs_blocks * l_scale_inv).astype(lhs_q_dtype)
    lhs_q = lhs_q.reshape(size_m, -1)[:, :size_k]
    l_scale = l_scale.squeeze(2).astype(acc_dtype)  # [size_m, num_l_blocks]

    # Match Pallas explicit cast when mixed-dtype matmul isn't natively supported
    if lhs_q_dtype == jnp.float8_e4m3fn and not jnp.issubdtype(
        rhs_proc.dtype, jnp.floating
    ):
      rhs_proc = rhs_proc.astype(lhs_q_dtype)
  else:
    lhs_q = lhs
    l_scale = None

  # Compute group interval offsets in LHS
  ends = jnp.cumsum(group_sizes, dtype=jnp.int32)
  starts = ends - group_sizes
  m_idx = jnp.arange(size_m, dtype=jnp.int32)

  # Compute global active range [compute_start, compute_end)
  g_ids = jnp.arange(size_lhs_group, dtype=jnp.int32) - go
  valid_g = (g_ids >= 0) & (g_ids < size_group) & (group_sizes > 0)
  any_valid = jnp.any(valid_g)
  first_valid_s = jnp.min(jnp.where(valid_g, starts, size_m))
  last_valid_e = jnp.max(jnp.where(valid_g, ends, 0))
  compute_start = jnp.where(any_valid, first_valid_s, 0)
  compute_end = jnp.where(any_valid, last_valid_e, 0)

  # Loop over local groups to compute matmul without materializing [G, M, N]
  def group_body(g, out_accum):
    lhs_g = g + go
    in_bounds = (lhs_g >= 0) & (lhs_g < size_lhs_group)
    safe_lhs_g = jnp.clip(lhs_g, 0, size_lhs_group - 1)
    g_size = jnp.where(in_bounds, group_sizes[safe_lhs_g], 0)
    s = jnp.where(in_bounds, starts[safe_lhs_g], 0)
    e = jnp.where(in_bounds, ends[safe_lhs_g], 0)

    def do_group(out_val):
      mask = (m_idx >= s) & (m_idx < e)  # [size_m]
      lhs_masked = jnp.where(mask[:, None], lhs_q, 0)
      w = rhs_proc[g]  # [size_k, size_n]

      if not should_quantize:
        if not dequant_after or num_blocks == 1:
          acc = jnp.matmul(
              lhs_masked, w, preferred_element_type=jnp.float32
          ).astype(acc_dtype)
          if dequant_after:
            acc = acc * rhs_scale_f32[g, 0].astype(acc_dtype)
        else:
          lhs_b = lhs_masked.reshape(size_m, num_blocks, rhs_qbs)
          w_b = w.reshape(num_blocks, rhs_qbs, size_n)
          block_acc = jnp.einsum(
              "mbk,bkn->bmn", lhs_b, w_b, preferred_element_type=jnp.float32
          ).astype(acc_dtype)
          r_scale = rhs_scale_f32[g].astype(acc_dtype)  # [num_blocks, 1, size_n]
          block_acc = block_acc * r_scale
          acc = jnp.sum(block_acc, axis=0, dtype=acc_dtype)
      else:
        pref_type = (
            jnp.float32
            if jnp.issubdtype(lhs_q_dtype, jnp.floating)
            else jnp.int32
        )
        q_block_size = 512
        pad_k = (-size_k) % q_block_size
        if pad_k > 0:
          lhs_m_pad = jnp.pad(lhs_masked, ((0, 0), (0, pad_k)))
          w_pad = jnp.pad(w, ((0, pad_k), (0, 0)))
        else:
          lhs_m_pad = lhs_masked
          w_pad = w
        num_l_blocks = lhs_m_pad.shape[1] // q_block_size
        lhs_b = lhs_m_pad.reshape(size_m, num_l_blocks, q_block_size)
        w_b = w_pad.reshape(num_l_blocks, q_block_size, size_n)

        block_acc = jnp.einsum(
            "mbk,bkn->bmn", lhs_b, w_b, preferred_element_type=pref_type
        ).astype(acc_dtype)
        # Apply LHS per-block scale: l_scale is [size_m, num_l_blocks]
        block_acc = block_acc * l_scale.T[:, :, None]

        if dequant_after:
          if num_blocks == 1:
            r_scale = rhs_scale_f32[g, 0:1].astype(acc_dtype)  # [1, 1, size_n]
          else:
            b_ids = (jnp.arange(num_l_blocks) * q_block_size) // rhs_qbs
            r_scale = rhs_scale_f32[g, b_ids].astype(
                acc_dtype
            )  # [num_l_blocks, 1, size_n]
          block_acc = block_acc * r_scale

        acc = jnp.sum(block_acc, axis=0, dtype=acc_dtype)

      if rhs_bias is not None:
        acc = acc + rhs_bias[g].astype(acc_dtype)

      acc_out = acc.astype(out_dtype)
      return jnp.where(mask[:, None], acc_out, out_val)

    return jax.lax.cond(g_size > 0, do_group, lambda x: x, out_accum)

  out = jnp.zeros((size_m, size_n), dtype=out_dtype)
  out = jax.lax.fori_loop(0, size_group, group_body, out)

  # Ensure rows outside [compute_start, compute_end) are strictly zeroed
  active_mask = (m_idx >= compute_start) & (m_idx < compute_end)
  out = jnp.where(active_mask[:, None], out, 0)
  return out
