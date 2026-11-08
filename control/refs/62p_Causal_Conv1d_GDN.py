import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp


CONFIGS = {
    "gdn_small": {
        "name": "gdn_small",
        "model": "GDN",
        "operator": "causal_conv1d_gated_delta_net",
        "num_tokens": 256,
        "n_kq": 2,
        "n_v": 8,
        "d_k": 128,
        "d_v": 128,
        "kernel_size": 4,
        "num_prefill_reqs": 2,
        "prefill_length": 128,
        "max_reqs": 32,
    },
    "gdn_prefill_medium": {
        "name": "gdn_prefill_medium",
        "model": "GDN",
        "operator": "causal_conv1d_gated_delta_net",
        "num_tokens": 2048,
        "n_kq": 2,
        "n_v": 8,
        "d_k": 128,
        "d_v": 128,
        "kernel_size": 4,
        "num_prefill_reqs": 4,
        "prefill_length": 512,
        "max_reqs": 64,
    },
    "gdn_prefill_large": {
        "name": "gdn_prefill_large",
        "model": "GDN",
        "operator": "causal_conv1d_gated_delta_net",
        "num_tokens": 8192,
        "n_kq": 2,
        "n_v": 8,
        "d_k": 128,
        "d_v": 128,
        "kernel_size": 4,
        "num_prefill_reqs": 8,
        "prefill_length": 1024,
        "max_reqs": 64,
    },
    "gdn_decode_b64": {
        "name": "gdn_decode_b64",
        "model": "GDN",
        "operator": "causal_conv1d_gated_delta_net",
        "num_tokens": 64,
        "n_kq": 2,
        "n_v": 8,
        "d_k": 128,
        "d_v": 128,
        "kernel_size": 4,
        "num_prefill_reqs": 0,
        "prefill_length": 0,
        "num_decode_reqs": 64,
        "max_reqs": 64,
    },
    "gdn_mixed_decode_prefill": {
        "name": "gdn_mixed_decode_prefill",
        "model": "GDN",
        "operator": "causal_conv1d_gated_delta_net",
        "num_tokens": 1056,
        "n_kq": 2,
        "n_v": 8,
        "d_k": 128,
        "d_v": 128,
        "kernel_size": 4,
        "num_prefill_reqs": 2,
        "prefill_length": 512,
        "num_decode_reqs": 32,
        "max_reqs": 64,
    },
}


CONFIG = CONFIGS["gdn_small"]


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns (qkv, b, a, conv_state, recurrent_state, conv_weight, conv_bias, a_log, dt_bias, query_start_loc, state_indices, distribution, seq_lens)."""
  cfg = (
      CONFIG
      if config is None
      else (CONFIGS[config] if isinstance(config, str) else config)
  )

  key = jax.random.key(42)
  k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = jax.random.split(key, 10)

  num_tokens = cfg["num_tokens"]
  n_kq = cfg["n_kq"]
  n_v = cfg["n_v"]
  d_k = cfg["d_k"]
  d_v = cfg["d_v"]
  kernel_size = cfg.get("kernel_size", 4)
  num_prefill_reqs = cfg.get("num_prefill_reqs", 0)
  prefill_length = cfg.get("prefill_length", 0)
  num_decode_reqs = cfg.get("num_decode_reqs", 0)
  max_reqs = cfg["max_reqs"]

  key_dim = n_kq * d_k
  val_dim = n_v * d_v
  mixed_dim = 2 * key_dim + val_dim

  qkv = jax.random.normal(k1, (num_tokens, mixed_dim), dtype=dtype)
  b = jax.random.normal(k2, (num_tokens, n_v), dtype=dtype)
  a = jax.random.normal(k3, (num_tokens, n_v), dtype=dtype)

  state_size = max_reqs + 1
  conv_state = jax.random.normal(
      k9, (state_size, kernel_size - 1, mixed_dim), dtype=dtype
  )
  recurrent_state = jax.random.normal(
      k10, (state_size, n_v, d_k, d_v), dtype=jnp.float32
  )
  recurrent_state = recurrent_state.at[0].set(0.0)

  conv_weight = jax.random.normal(k4, (mixed_dim, 1, kernel_size), dtype=dtype)
  conv_bias = jax.random.normal(k5, (mixed_dim,), dtype=dtype)

  a_log = jax.random.normal(k6, (n_v,), dtype=jnp.float32)
  dt_bias = jax.random.normal(k7, (n_v,), dtype=jnp.float32)

  lengths = [1] * num_decode_reqs + [prefill_length] * num_prefill_reqs
  num_seqs = len(lengths)
  lengths_with_past = [1 + 128] * num_decode_reqs + [
      prefill_length
  ] * num_prefill_reqs
  query_start_loc = jnp.cumsum(jnp.array([0] + lengths, dtype=jnp.int32))
  last_val = query_start_loc[-1]
  query_start_loc = jnp.pad(
      query_start_loc,
      (0, max_reqs + 1 - len(query_start_loc)),
      constant_values=last_val,
  )

  available_indices = jax.random.permutation(
      k8, jnp.arange(1, max_reqs + 1, dtype=jnp.int32)
  )
  valid_state_indices = available_indices[:num_seqs]
  state_indices = jnp.pad(valid_state_indices, (0, max_reqs - num_seqs))

  distribution = jnp.array(
      [num_decode_reqs, num_seqs, num_seqs], dtype=jnp.int32
  )
  seq_lens = jnp.pad(
      jnp.array(lengths_with_past, dtype=jnp.int32), (0, max_reqs - num_seqs)
  )

  return (
      qkv,
      b,
      a,
      conv_state,
      recurrent_state,
      conv_weight,
      conv_bias,
      a_log,
      dt_bias,
      query_start_loc,
      state_indices,
      distribution,
      seq_lens,
  )


import jax
import jax.numpy as jnp


def _cdiv(a, b):
  return (a + b - 1) // b


def _l2_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
  norm = jnp.sqrt(jnp.sum(x * x, axis=-1, keepdims=True, dtype=x.dtype) + eps)
  return x / norm


def _invert_triangular_matrix(t: jax.Array, block_size: int = 16) -> jax.Array:
  out_dtype = t.dtype
  chunk = t.shape[-1]
  block_size = min(block_size, chunk)
  num_blocks = chunk // block_size

  def local_forward_sub(t_mat: jax.Array, b_mat: jax.Array) -> jax.Array:
    x_list = []
    for i in range(block_size):
      b_i = b_mat[..., i, :]
      if i == 0:
        x_i = b_i
      else:
        stacked_x = jnp.stack(x_list, axis=-2)
        all_prev_t = t_mat[..., i, :i]
        prev_sum = jnp.sum(all_prev_t[..., None] * stacked_x, axis=-2)
        x_i = b_i - prev_sum
      x_list.append(x_i)
    return jnp.stack(x_list, axis=-2)

  x_blocks = []
  iota_r = jax.lax.broadcasted_iota(jnp.int32, t.shape, t.ndim - 2)
  iota_c = jax.lax.broadcasted_iota(jnp.int32, t.shape, t.ndim - 1)
  identity_mask = jnp.where(iota_r == iota_c, 1.0, 0.0)
  batch_dims = tuple(range(t.ndim - 2))

  for i in range(num_blocks):
    start, end = i * block_size, (i + 1) * block_size
    e_block = identity_mask[..., start:end, :]

    if i == 0:
      target_b = e_block
    else:
      interaction_t = t[..., start:end, :start]
      solved_x = jnp.concatenate(x_blocks, axis=-2)
      prev_sum = jax.lax.dot_general(
          interaction_t,
          solved_x,
          dimension_numbers=(
              ((t.ndim - 1,), (t.ndim - 2,)),
              (batch_dims, batch_dims),
          ),
          preferred_element_type=jnp.float32,
      )
      target_b = e_block - prev_sum

    local_t = t[..., start:end, start:end].astype(jnp.float32)
    x_block = local_forward_sub(local_t, target_b)
    x_blocks.append(x_block.astype(out_dtype))

  return jnp.concatenate(x_blocks, axis=-2)


def _causal_conv1d(
    real_sizes: jax.Array,  # [seq]
    lhs: jax.Array,  # [seq, prev_kernel_size + chunk, dim_size]
    conv_weight: jax.Array,  # [kernel_size, dim_size]
    conv_bias: jax.Array | None,  # [dim_size]
    chunk_size: int,
    kernel_size: int,
) -> tuple[jax.Array, jax.Array]:
  seq_tile_size, _, dim_size = lhs.shape
  prev_kernel_size = kernel_size - 1

  out_list = []
  for c_idx in range(chunk_size):
    out = jnp.zeros((seq_tile_size, dim_size), jnp.float32)
    start_idx = c_idx
    for k in range(kernel_size):
      lhs_curr = lhs[:, start_idx + k]
      out += lhs_curr * conv_weight[k : k + 1]
    if conv_bias is not None:
      out += conv_bias.reshape(1, -1)
    out_list.append(out)

  new_conv_state = lhs[:, 1:kernel_size]
  real_sizes_b = real_sizes.reshape(-1, 1, 1)
  for c_idx in range(2, chunk_size + 1):
    row_end = c_idx + prev_kernel_size
    new_conv_state = jnp.where(
        c_idx == real_sizes_b,
        lhs[:, c_idx:row_end],
        new_conv_state,
    )

  return jnp.stack(out_list, axis=1), new_conv_state


def _recurrent_gdn_step(
    real_sizes: jax.Array,  # [B]
    qkv_act: jax.Array,  # [B, 1, dim]
    b_act: jax.Array,  # [B, 1, n_v]
    a_act: jax.Array,  # [B, 1, n_v]
    state_prev: jax.Array,  # [B, n_v, d_k, d_v]
    a_log: jax.Array,  # [n_v]
    dt_bias: jax.Array,  # [n_v]
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
) -> tuple[jax.Array, jax.Array]:
  B = qkv_act.shape[0]
  v_per_kq = n_v // n_kq
  k_offset = n_kq * d_k
  v_offset = 2 * n_kq * d_k

  q = qkv_act[:, 0, :k_offset].reshape(B, n_kq, d_k)
  k = qkv_act[:, 0, k_offset:v_offset].reshape(B, n_kq, d_k)
  v = qkv_act[:, 0, v_offset:].reshape(B, n_v, d_v)

  mask = (real_sizes > 0).reshape(B, 1, 1)
  q = jnp.where(mask, q.astype(jnp.float32), 0.0)
  k = jnp.where(mask, k.astype(jnp.float32), 0.0)
  v = jnp.where(mask, v.astype(jnp.float32), 0.0)

  q = _l2_norm(q) * (d_k**-0.5)
  k = _l2_norm(k)

  b_val = b_act[:, 0, :].astype(jnp.float32)
  a_val = a_act[:, 0, :].astype(jnp.float32)

  beta = jax.nn.sigmoid(b_val)
  gating_log = -jnp.exp(a_log.astype(jnp.float32)) * jax.nn.softplus(
      a_val + dt_bias.astype(jnp.float32)
  )

  mask_2d = (real_sizes > 0).reshape(B, 1)
  beta = jnp.where(mask_2d, beta, 0.0)
  gating_log = jnp.where(mask_2d, gating_log, 0.0)
  gating = jnp.exp(gating_log)

  q_curr = jnp.repeat(q, v_per_kq, axis=1)[:, :, None, :]  # [B, n_v, 1, d_k]
  k_curr = jnp.repeat(k, v_per_kq, axis=1)[:, :, None, :]  # [B, n_v, 1, d_k]
  k_curr_t = jnp.repeat(k, v_per_kq, axis=1)[:, :, :, None]  # [B, n_v, d_k, 1]
  v_curr = v[:, :, None, :]  # [B, n_v, 1, d_v]

  beta_curr = beta.reshape(B, n_v, 1, 1)
  gating_curr = gating.reshape(B, n_v, 1, 1)

  state_updated = state_prev * gating_curr

  kq_merged = jnp.concatenate([k_curr, q_curr], axis=2)  # [B, n_v, 2, d_k]
  kq_proj = jax.lax.dot_general(
      kq_merged,
      state_updated,
      dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1))),
      preferred_element_type=jnp.float32,
  )  # [B, n_v, 2, d_v]

  v_updated = kq_proj[:, :, :1, :]
  out_decayed = kq_proj[:, :, 1:, :]

  v_diff = v_curr - v_updated
  v_new = beta_curr * v_diff

  qk_dot = jnp.sum(q_curr * k_curr, axis=-1, keepdims=True)
  out = out_decayed + qk_dot * v_new  # [B, n_v, 1, d_v]

  state_new = k_curr_t * v_new
  new_recurrent_state = state_updated + state_new

  out = out.swapaxes(1, 2)  # [B, 1, n_v, d_v]
  return out, new_recurrent_state


def _chunked_gdn_step(
    real_sizes: jax.Array,  # [1]
    qkv_act: jax.Array,  # [1, C, dim]
    b_act: jax.Array,  # [1, C, n_v]
    a_act: jax.Array,  # [1, C, n_v]
    state_prev: jax.Array,  # [1, n_v, d_k, d_v]
    a_log: jax.Array,  # [n_v]
    dt_bias: jax.Array,  # [n_v]
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
  v_per_kq = n_v // n_kq
  k_offset = n_kq * d_k
  v_offset = 2 * n_kq * d_k

  # Reshape and transpose to [heads, C, head_dim]
  q_large = (
      qkv_act[0, :, :k_offset].reshape(chunk_size, n_kq, d_k).swapaxes(0, 1)
  )
  k_large = (
      qkv_act[0, :, k_offset:v_offset]
      .reshape(chunk_size, n_kq, d_k)
      .swapaxes(0, 1)
  )
  v_large = (
      qkv_act[0, :, v_offset:].reshape(chunk_size, n_v, d_v).swapaxes(0, 1)
  )

  iota = jnp.arange(chunk_size, dtype=jnp.int32).reshape(1, chunk_size, 1)
  mask = iota < real_sizes.reshape(1, 1, 1)

  q_large = jnp.where(mask, q_large.astype(jnp.float32), 0.0)
  k_large = jnp.where(mask, k_large.astype(jnp.float32), 0.0)
  v_large = jnp.where(mask, v_large.astype(jnp.float32), 0.0)

  q_large = _l2_norm(q_large) * (d_k**-0.5)
  k_large = _l2_norm(k_large)

  b_large = b_act[0].astype(jnp.float32)  # [C, n_v]
  a_large = a_act[0].astype(jnp.float32)  # [C, n_v]

  beta = jax.nn.sigmoid(b_large)
  gating_log = -jnp.exp(a_log.astype(jnp.float32))[None, :] * jax.nn.softplus(
      a_large + dt_bias.astype(jnp.float32)[None, :]
  )

  mask_2d = iota[0] < real_sizes.reshape(1, 1)  # [C, 1]
  beta = jnp.where(mask_2d, beta, 0.0)
  gating_log = jnp.where(mask_2d, gating_log, 0.0)

  q_repeat = jnp.repeat(q_large, v_per_kq, axis=0)  # [n_v, C, d_k]
  k_repeat = jnp.repeat(k_large, v_per_kq, axis=0)  # [n_v, C, d_k]

  g_cum_sum_list = [gating_log[:1, :]]
  for row in range(1, chunk_size):
    g_cum_sum_list.append(g_cum_sum_list[-1] + gating_log[row : row + 1, :])
  g_cum_sum_log = jnp.concatenate(g_cum_sum_list, axis=0)  # [C, n_v]
  g_cum_sum_log = g_cum_sum_log.T[:, :, None]  # [n_v, C, 1]
  beta_large = beta.T[:, :, None]  # [n_v, C, 1]

  g_cum_sum_log_t = jnp.swapaxes(g_cum_sum_log, 1, 2)  # [n_v, 1, C]
  g_cum_sum_diff_log = g_cum_sum_log - g_cum_sum_log_t  # [n_v, C, C]
  gating_map = jnp.exp(g_cum_sum_diff_log)
  gating_backward = jnp.exp(-g_cum_sum_diff_log[..., -1:])  # [n_v, C, 1]
  gating_forward = jnp.exp(g_cum_sum_log)  # [n_v, C, 1]
  gating_last = gating_forward[:, -1:]  # [n_v, 1, 1]

  iota_r = jax.lax.broadcasted_iota(jnp.int32, gating_map.shape, 1)
  iota_c = jax.lax.broadcasted_iota(jnp.int32, gating_map.shape, 2)
  identity_mask = iota_r == iota_c
  strictly_lower_mask = iota_r > iota_c
  lower_mask = iota_r >= iota_c

  gating_map_masked = jnp.where(strictly_lower_mask, gating_map, 0.0)
  k_beta_repeat = k_repeat * beta_large

  beta_k_k_t = jax.lax.dot(
      k_beta_repeat,
      k_repeat,
      dimension_numbers=(((2,), (2,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )
  gating_beta_k_k_t = gating_map_masked * beta_k_k_t
  t = jnp.where(identity_mask, 1.0, gating_beta_k_k_t)

  t_inv = _invert_triangular_matrix(t)

  v_beta_large = v_large * beta_large
  k_beta_gating = k_beta_repeat * gating_forward
  merged_v_k = jnp.concatenate([v_beta_large, k_beta_gating], axis=-1)
  merged_uw = jax.lax.dot(
      t_inv,
      merged_v_k,
      dimension_numbers=(((2,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )
  u, w = jnp.split(merged_uw, [d_v], axis=-1)

  q_large_gating = q_repeat * gating_forward
  merged_w_q = jnp.concatenate([w, q_large_gating], axis=1)
  merged_ws_out_updated = jax.lax.dot(
      merged_w_q,
      state_prev[0],
      dimension_numbers=(((2,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )
  ws, out_updated = jnp.split(merged_ws_out_updated, 2, axis=1)
  u_ws = u - ws

  k_repeat_gating = k_repeat * gating_backward
  state_new = jax.lax.dot(
      k_repeat_gating,
      u_ws,
      dimension_numbers=(((1,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )
  state_updated = state_prev[0] * gating_last
  state = state_updated + state_new

  out_qk = jax.lax.dot(
      q_large,
      k_large,
      dimension_numbers=(((2,), (2,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )
  out_qk = jnp.repeat(out_qk, v_per_kq, axis=0)
  out_qk *= gating_map
  out_qk = jnp.where(lower_mask, out_qk, 0.0)

  out_new = jax.lax.dot(
      out_qk,
      u_ws,
      dimension_numbers=(((2,), (1,)), ((0,), (0,))),
      preferred_element_type=jnp.float32,
  )
  out = out_updated + out_new  # [n_v, C, d_v]

  out = out.swapaxes(0, 1)[None, ...]  # [1, C, n_v, d_v]
  state = state[None, ...]  # [1, n_v, d_k, d_v]
  return out, state


def workload(
    qkv,
    b,
    a,
    conv_state,
    recurrent_state,
    conv_weight,
    conv_bias,
    a_log,
    dt_bias,
    query_start_loc,
    state_indices,
    distribution,
    seq_lens,
):
  act_in_dtype = qkv.dtype
  act_out_dtype = qkv.dtype
  conv_out_dtype = conv_state.dtype
  recurrent_out_dtype = recurrent_state.dtype

  n_v = b.shape[1]
  n_kq = 2
  d_k = 128
  d_v = 128
  kernel_size = 4
  prev_kernel_size = kernel_size - 1
  decode_tile_size = 4
  mixed_tile_size = 64

  num_seqs = state_indices.size
  batch_size, dim = qkv.shape
  packing = 4 // act_in_dtype.itemsize
  padded_batch_size = _cdiv(batch_size, packing) * packing

  decode_tile_size = max(1, min(decode_tile_size, num_seqs))
  mixed_tile_size = max(1, min(mixed_tile_size, batch_size))

  qkv_f32 = jnp.pad(
      qkv.astype(jnp.float32), ((0, padded_batch_size - batch_size), (0, 0))
  )
  b_f32 = jnp.pad(
      b.astype(jnp.float32), ((0, padded_batch_size - batch_size), (0, 0))
  )
  a_f32 = jnp.pad(
      a.astype(jnp.float32), ((0, padded_batch_size - batch_size), (0, 0))
  )

  conv_state_f32 = conv_state.astype(jnp.float32)
  conv_w = conv_weight.swapaxes(0, 2).astype(jnp.float32)[
      :, 0, :
  ]  # [kernel_size, dim]
  conv_b = conv_bias.astype(jnp.float32) if conv_bias is not None else None

  out_act_init = jnp.zeros((padded_batch_size, n_v, d_v), dtype=act_out_dtype)

  # ============================================================================
  # Phase 1: Decode pass over sequences [0, distribution[0])
  # ============================================================================
  def _run_decode(in_conv_state, in_rec_state, in_act):
    end_seq = distribution[0]
    all_seqs = jnp.arange(num_seqs, dtype=jnp.int32)
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    is_valid_seqs = all_seqs < end_seq
    has_initial_state = (seq_lens - query_lens) > 0
    all_valid_seqs = jnp.where(is_valid_seqs, all_seqs, 0)
    r_sizes = jnp.where(is_valid_seqs, 1, 0)
    num_tiles = _cdiv(end_seq, decode_tile_size)

    def decode_tile_step(p_id, carry):
      curr_act, curr_conv, curr_rec = carry
      idx_offset = p_id * decode_tile_size + jnp.arange(
          decode_tile_size, dtype=jnp.int32
      )
      s_idx = all_valid_seqs[idx_offset]
      r_base = all_valid_seqs[idx_offset]
      r_size = r_sizes[idx_offset]
      is_last = is_valid_seqs[idx_offset]
      has_init = has_initial_state[s_idx]
      st_idx = state_indices[s_idx]

      qkv_tile = qkv_f32[r_base][:, None, :]  # [tile, 1, dim]
      b_tile = b_f32[r_base][:, None, :]  # [tile, 1, n_v]
      a_tile = a_f32[r_base][:, None, :]  # [tile, 1, n_v]

      prev_conv = jnp.where(
          has_init[:, None, None], curr_conv[st_idx], 0.0
      )  # [tile, prev_k, dim]
      prev_rec = jnp.where(
          has_init[:, None, None, None], curr_rec[st_idx], 0.0
      )  # [tile, n_v, d_k, d_v]

      qkv_in = jnp.concatenate([prev_conv, qkv_tile], axis=1)
      qkv_out, new_conv = _causal_conv1d(
          r_size, qkv_in, conv_w, conv_b, chunk_size=1, kernel_size=kernel_size
      )
      qkv_out = jax.nn.silu(qkv_out)

      out_tile, new_rec = _recurrent_gdn_step(
          r_size,
          qkv_out,
          b_tile,
          a_tile,
          prev_rec,
          a_log,
          dt_bias,
          n_kq=n_kq,
          n_v=n_v,
          d_k=d_k,
          d_v=d_v,
      )
      out_tile = out_tile.astype(act_out_dtype)[:, 0, :, :]  # [tile, n_v, d_v]
      new_rec = new_rec.astype(curr_rec.dtype)

      valid_r = r_size > 0
      curr_act = curr_act.at[r_base].set(
          jnp.where(valid_r[:, None, None], out_tile, curr_act[r_base])
      )
      curr_conv = curr_conv.at[st_idx].set(
          jnp.where(is_last[:, None, None], new_conv, curr_conv[st_idx])
      )
      curr_rec = curr_rec.at[st_idx].set(
          jnp.where(is_last[:, None, None, None], new_rec, curr_rec[st_idx])
      )
      return curr_act, curr_conv, curr_rec

    return jax.lax.fori_loop(
        0, num_tiles, decode_tile_step, (in_act, in_conv_state, in_rec_state)
    )

  out_act, out_conv_state, out_recurrent_state = jax.lax.cond(
      distribution[0] > 0,
      lambda: _run_decode(conv_state_f32, recurrent_state, out_act_init),
      lambda: (out_act_init, conv_state_f32, recurrent_state),
  )

  # ============================================================================
  # Phase 2: Prefill/mixed pass over sequences [distribution[0], distribution[-1])
  # ============================================================================
  def _run_prefill(in_conv_state, in_rec_state, in_act):
    start_seq = distribution[0]
    end_seq = distribution[-1]
    all_seqs = jnp.arange(num_seqs, dtype=jnp.int32)
    all_tokens = jnp.arange(padded_batch_size, dtype=jnp.int32)

    q_start = jnp.roll(query_start_loc, shift=-start_seq)
    s_lens = jnp.roll(seq_lens, shift=-start_seq)
    st_indices = jnp.roll(state_indices, shift=-start_seq)

    query_lens = q_start[1:] - q_start[:-1]
    n_active_seqs = end_seq - start_seq
    query_lens = jnp.where(all_seqs < n_active_seqs, query_lens, 0)

    s_idx_to_num_tiles = _cdiv(query_lens, mixed_tile_size)
    s_idx_to_start_p_id = jnp.cumulative_sum(
        s_idx_to_num_tiles, include_initial=True
    )
    p_id_to_s_idx = jnp.repeat(
        all_seqs, s_idx_to_num_tiles, total_repeat_length=padded_batch_size
    )
    p_id_to_t_id = all_tokens - s_idx_to_start_p_id[p_id_to_s_idx]
    p_id_to_r_base = q_start[p_id_to_s_idx] + p_id_to_t_id * mixed_tile_size
    p_id_to_r_size = jnp.minimum(
        q_start[p_id_to_s_idx + 1] - p_id_to_r_base, mixed_tile_size
    )

    has_initial_state = (s_lens - query_lens) > 0
    p_id_is_first_tile = p_id_to_t_id == 0
    p_id_is_last_tile = p_id_to_t_id == (s_idx_to_num_tiles[p_id_to_s_idx] - 1)
    num_tiles = s_idx_to_num_tiles.sum()

    qkv_padded = jnp.pad(qkv_f32, ((0, mixed_tile_size), (0, 0)))
    b_padded = jnp.pad(b_f32, ((0, mixed_tile_size), (0, 0)))
    a_padded = jnp.pad(a_f32, ((0, mixed_tile_size), (0, 0)))
    act_padded = jnp.pad(in_act, ((0, mixed_tile_size), (0, 0), (0, 0)))

    init_carry_conv = jnp.zeros((1, prev_kernel_size, dim), dtype=jnp.float32)
    init_carry_rec = jnp.zeros((1, n_v, d_k, d_v), dtype=jnp.float32)

    def prefill_tile_step(p_id, carry):
      curr_act, curr_conv, curr_rec, carry_conv, carry_rec = carry
      s_idx = p_id_to_s_idx[p_id]
      r_base = p_id_to_r_base[p_id]
      r_size = p_id_to_r_size[p_id]
      is_first = p_id_is_first_tile[p_id]
      is_last = p_id_is_last_tile[p_id]
      has_init = has_initial_state[s_idx]
      st_idx = st_indices[s_idx]

      qkv_tile = jax.lax.dynamic_slice(
          qkv_padded, (r_base, 0), (mixed_tile_size, dim)
      )[None, ...]
      b_tile = jax.lax.dynamic_slice(
          b_padded, (r_base, 0), (mixed_tile_size, n_v)
      )[None, ...]
      a_tile = jax.lax.dynamic_slice(
          a_padded, (r_base, 0), (mixed_tile_size, n_v)
      )[None, ...]

      hbm_conv = curr_conv[st_idx][None, ...]
      prev_conv = jnp.where(has_init, hbm_conv, 0.0)
      prev_conv = jnp.where(is_first, prev_conv, carry_conv)

      hbm_rec = curr_rec[st_idx][None, ...].astype(jnp.float32)
      prev_rec = jnp.where(has_init, hbm_rec, 0.0)
      prev_rec = jnp.where(is_first, prev_rec, carry_rec)

      r_size_arr = r_size[None]
      qkv_in = jnp.concatenate([prev_conv, qkv_tile], axis=1)
      qkv_out, new_conv = _causal_conv1d(
          r_size_arr,
          qkv_in,
          conv_w,
          conv_b,
          chunk_size=mixed_tile_size,
          kernel_size=kernel_size,
      )
      qkv_out = jax.nn.silu(qkv_out)

      if mixed_tile_size == 1:
        out_tile, new_rec = _recurrent_gdn_step(
            r_size_arr,
            qkv_out,
            b_tile,
            a_tile,
            prev_rec,
            a_log,
            dt_bias,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
        )
      else:
        out_tile, new_rec = _chunked_gdn_step(
            r_size_arr,
            qkv_out,
            b_tile,
            a_tile,
            prev_rec,
            a_log,
            dt_bias,
            n_kq=n_kq,
            n_v=n_v,
            d_k=d_k,
            d_v=d_v,
            chunk_size=mixed_tile_size,
        )

      out_tile = out_tile[0].astype(act_out_dtype)  # [C, n_v, d_v]
      existing_slice = jax.lax.dynamic_slice(
          curr_act, (r_base, 0, 0), (mixed_tile_size, n_v, d_v)
      )
      row_mask = (jnp.arange(mixed_tile_size, dtype=jnp.int32) < r_size)[
          :, None, None
      ]
      updated_slice = jnp.where(row_mask, out_tile, existing_slice)
      curr_act = jax.lax.dynamic_update_slice(
          curr_act, updated_slice, (r_base, 0, 0)
      )

      curr_conv = curr_conv.at[st_idx].set(
          jnp.where(is_last, new_conv[0], curr_conv[st_idx])
      )
      curr_rec = curr_rec.at[st_idx].set(
          jnp.where(
              is_last, new_rec[0].astype(curr_rec.dtype), curr_rec[st_idx]
          )
      )

      return curr_act, curr_conv, curr_rec, new_conv, new_rec

    act_padded, out_conv, out_rec, _, _ = jax.lax.fori_loop(
        0,
        num_tiles,
        prefill_tile_step,
        (act_padded, in_conv_state, in_rec_state, init_carry_conv, init_carry_rec),
    )
    return act_padded[:padded_batch_size], out_conv, out_rec

  out_act, out_conv_state, out_recurrent_state = jax.lax.cond(
      distribution[-1] > distribution[0],
      lambda: _run_prefill(out_conv_state, out_recurrent_state, out_act),
      lambda: (out_act, out_conv_state, out_recurrent_state),
  )

  out_act = out_act.reshape(padded_batch_size, -1)[:batch_size]
  out_conv_state = out_conv_state.astype(conv_out_dtype)
  out_recurrent_state = out_recurrent_state.astype(recurrent_out_dtype)

  return (out_conv_state, out_recurrent_state), out_act
