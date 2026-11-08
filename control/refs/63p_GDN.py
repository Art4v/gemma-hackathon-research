import jax
import jax.numpy as jnp
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp


CONFIGS = {
    'gdn_small': {
        'name': 'gdn_small',
        'model': 'GDN',
        'operator': 'gated_delta_net',
        'num_tokens': 256,
        'n_kq': 2,
        'n_v': 8,
        'd_k': 128,
        'd_v': 128,
        'num_prefill_reqs': 2,
        'prefill_length': 128,
        'max_reqs': 32,
    },
    'gdn_prefill_medium': {
        'name': 'gdn_prefill_medium',
        'model': 'GDN',
        'operator': 'gated_delta_net',
        'num_tokens': 2048,
        'n_kq': 2,
        'n_v': 8,
        'd_k': 128,
        'd_v': 128,
        'num_prefill_reqs': 4,
        'prefill_length': 512,
        'max_reqs': 64,
    },
    'gdn_prefill_large': {
        'name': 'gdn_prefill_large',
        'model': 'GDN',
        'operator': 'gated_delta_net',
        'num_tokens': 8192,
        'n_kq': 2,
        'n_v': 8,
        'd_k': 128,
        'd_v': 128,
        'num_prefill_reqs': 8,
        'prefill_length': 1024,
        'max_reqs': 64,
    },
    'gdn_decode_b64': {
        'name': 'gdn_decode_b64',
        'model': 'GDN',
        'operator': 'gated_delta_net',
        'num_tokens': 64,
        'n_kq': 2,
        'n_v': 8,
        'd_k': 128,
        'd_v': 128,
        'num_prefill_reqs': 0,
        'prefill_length': 0,
        'num_decode_reqs': 64,
        'max_reqs': 64,
    },
    'gdn_mixed_decode_prefill': {
        'name': 'gdn_mixed_decode_prefill',
        'model': 'GDN',
        'operator': 'gated_delta_net',
        'num_tokens': 1056,
        'n_kq': 2,
        'n_v': 8,
        'd_k': 128,
        'd_v': 128,
        'num_prefill_reqs': 2,
        'prefill_length': 512,
        'num_decode_reqs': 32,
        'max_reqs': 64,
    },
}


CONFIG = CONFIGS['gdn_small']


def create_inputs(dtype=jnp.bfloat16, config=None):
  """Returns (mixed_qkv, b, a, recurrent_state, A_log, dt_bias, query_start_loc, state_indices, distribution, has_initial_state)."""
  cfg = (
      CONFIG
      if config is None
      else (CONFIGS[config] if isinstance(config, str) else config)
  )

  key = jax.random.key(42)
  k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

  num_tokens = cfg['num_tokens']
  n_kq = cfg['n_kq']
  n_v = cfg['n_v']
  d_k = cfg['d_k']
  d_v = cfg['d_v']
  num_prefill_reqs = cfg.get('num_prefill_reqs', 0)
  prefill_length = cfg.get('prefill_length', 0)
  num_decode_reqs = cfg.get('num_decode_reqs', 0)
  max_reqs = cfg['max_reqs']

  key_dim = n_kq * d_k
  val_dim = n_v * d_v
  mixed_dim = 2 * key_dim + val_dim

  mixed_qkv = jax.random.normal(k1, (num_tokens, mixed_dim), dtype=dtype)
  b = jax.random.normal(k2, (num_tokens, n_v), dtype=dtype)
  a = jax.random.normal(k3, (num_tokens, n_v), dtype=dtype)

  state_size = max_reqs + 1
  recurrent_state = jax.random.normal(
      k7, (state_size, n_v, d_k, d_v), dtype=jnp.float32
  )
  # Slot 0 is the null block that `state_indices` is zero-padded with
  # for the inactive request slots. Implementations differ on whether
  # they touch it: the ref zeroes every slot named in `state_indices`
  # whose `has_initial_state` is False (which includes the padding),
  # while the kernels skip inactive requests entirely. Seed it with
  # zeros so the null block's final contents are well defined either
  # way and don't show up as a spurious mismatch.
  recurrent_state = recurrent_state.at[0].set(0.0)

  A_log = jax.random.normal(k4, (n_v,), dtype=jnp.float32)
  dt_bias = jax.random.normal(k5, (n_v,), dtype=jnp.float32)

  lengths = [1] * num_decode_reqs + [prefill_length] * num_prefill_reqs
  num_seqs = len(lengths)
  query_start_loc = jnp.cumsum(jnp.array([0] + lengths, dtype=jnp.int32))
  last_val = query_start_loc[-1]
  query_start_loc = jnp.pad(
      query_start_loc,
      (0, max_reqs + 1 - len(query_start_loc)),
      constant_values=last_val,
  )

  available_indices = jax.random.permutation(
      k6, jnp.arange(1, max_reqs + 1, dtype=jnp.int32)
  )
  valid_state_indices = available_indices[:num_seqs]
  state_indices = jnp.pad(valid_state_indices, (0, max_reqs - num_seqs))

  distribution = jnp.array(
      [num_decode_reqs, num_seqs, num_seqs], dtype=jnp.int32
  )
  has_initial_state = jnp.pad(
      jnp.array(
          [True] * num_decode_reqs + [False] * num_prefill_reqs,
          dtype=jnp.bool_,
      ),
      (0, max_reqs - num_seqs),
  )

  return (
      mixed_qkv,
      b,
      a,
      recurrent_state,
      A_log,
      dt_bias,
      query_start_loc,
      state_indices,
      distribution,
      has_initial_state,
  )


import jax
import jax.numpy as jnp


def workload(
    mixed_qkv,
    b,
    a,
    recurrent_state,
    A_log,
    dt_bias,
    query_start_loc,
    state_indices,
    distribution,
    has_initial_state,
):
  """Plain JAX implementation of ragged Gated Delta Net (GDN)."""
  num_tokens = mixed_qkv.shape[0]
  out_dtype = mixed_qkv.dtype
  state_dtype = recurrent_state.dtype

  _, n_v, d_k, d_v = recurrent_state.shape
  val_dim = n_v * d_v
  key_dim = (mixed_qkv.shape[1] - val_dim) // 2
  n_kq = key_dim // d_k
  repeat_factor = n_v // n_kq
  scale = d_k**-0.5

  # Extract and reshape q, k, v
  q = mixed_qkv[:, :key_dim].reshape(num_tokens, n_kq, d_k)
  k = mixed_qkv[:, key_dim : 2 * key_dim].reshape(num_tokens, n_kq, d_k)
  v = mixed_qkv[:, 2 * key_dim :].reshape(num_tokens, n_v, d_v)

  # Apply fused SiLU
  q = jax.nn.silu(q.astype(jnp.float32))
  k = jax.nn.silu(k.astype(jnp.float32))
  v = jax.nn.silu(v.astype(jnp.float32))

  # Apply L2 normalization and scaling to q, k
  q = q * jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6) * scale
  k = k * jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)

  # GQA: repeat q and k heads to match n_v
  if repeat_factor > 1:
    q = jnp.repeat(q, repeat_factor, axis=1)
    k = jnp.repeat(k, repeat_factor, axis=1)

  # Compute gating and beta across all tokens
  a_val = jnp.exp(A_log.astype(jnp.float32))
  g_val = a.astype(jnp.float32) + dt_bias.astype(jnp.float32)[None, :]
  decay = jnp.exp(-a_val[None, :] * jax.nn.softplus(g_val))  # [T, n_v]
  beta = jax.nn.sigmoid(b.astype(jnp.float32))  # [T, n_v]

  decode_end = distribution[0]
  total_seqs = distribution[2]

  # Map each token index t in [0, num_tokens) to its sequence index in [0, max_reqs)
  t_arange = jnp.arange(num_tokens, dtype=jnp.int32)
  seq_idx_per_token = (
      jnp.sum(query_start_loc[1:, None] <= t_arange[None, :], axis=0)
  ).astype(jnp.int32)
  is_seq_start_arr = t_arange == query_start_loc[seq_idx_per_token]

  # Count how many valid tokens belong to active sequences
  total_valid_tokens = jnp.where(
      total_seqs > decode_end, query_start_loc[total_seqs], decode_end
  )

  # Scan over tokens to compute recurrent outputs
  def scan_step(carry, xs):
    h, curr_seq = carry
    t, q_t, k_t, v_t, decay_t, beta_t, seq_idx, is_start = xs

    is_decode = t < decode_end
    actual_seq = jnp.where(is_decode, t, seq_idx)
    start_new = is_decode | is_start

    def load_new_h(_):
      si = state_indices[actual_seq]
      h_loaded = recurrent_state[si].astype(jnp.float32)
      if has_initial_state is not None:
        use_init = is_decode | (has_initial_state[actual_seq] != 0)
        h_loaded = jnp.where(use_init, h_loaded, 0.0)
      return h_loaded

    h = jax.lax.cond(start_new, load_new_h, lambda _: h, None)

    h_pre = h * decay_t[:, None, None]
    kh = jnp.einsum('hk,hkv->hv', k_t, h_pre)
    v_diff = v_t - kh
    b_v = beta_t[:, None] * v_diff

    o_step1 = jnp.einsum('hk,hkv->hv', q_t, h_pre)
    qk_dot = jnp.sum(q_t * k_t, axis=-1, keepdims=True)
    o_t = o_step1 + qk_dot * b_v
    h_new = h_pre + k_t[:, :, None] * b_v[:, None, :]

    is_valid = t < total_valid_tokens
    h_out = jnp.where(is_valid, h_new, h)
    o_out = jnp.where(
        is_valid, o_t.astype(out_dtype), v_t.astype(out_dtype)
    )
    return (h_out, actual_seq), (o_out, h_out)

  init_carry = (
      jnp.zeros((n_v, d_k, d_v), dtype=jnp.float32),
      jnp.int32(-1),
  )
  xs = (
      t_arange,
      q,
      k,
      v,
      decay,
      beta,
      seq_idx_per_token,
      is_seq_start_arr,
  )
  _, (o_all, h_all) = jax.lax.scan(scan_step, init_carry, xs)

  # Preserve untouched raw `v` rows if total_valid_tokens < num_tokens
  raw_v = mixed_qkv[:, 2 * key_dim :].reshape(num_tokens, n_v, d_v)
  valid_mask = (t_arange < total_valid_tokens)[:, None, None]
  o_all = jnp.where(valid_mask, o_all, raw_v)

  # Update recurrent_state at the last token of each active sequence
  max_reqs = state_indices.shape[0]
  req_arange = jnp.arange(max_reqs, dtype=jnp.int32)
  is_decode_req = req_arange < decode_end
  is_prefill_req = (req_arange >= decode_end) & (req_arange < total_seqs)

  seq_len = query_start_loc[req_arange + 1] - query_start_loc[req_arange]
  is_active_req = is_decode_req | (is_prefill_req & (seq_len > 0))

  last_token_idx = jnp.where(
      is_decode_req, req_arange, query_start_loc[req_arange + 1] - 1
  )
  safe_last_token_idx = jnp.clip(last_token_idx, 0, num_tokens - 1)
  final_h_per_req = h_all[safe_last_token_idx].astype(state_dtype)

  target_slots = jnp.where(
      is_active_req, state_indices, recurrent_state.shape[0]
  )
  new_recurrent_state = recurrent_state.at[target_slots].set(
      final_h_per_req, mode='drop'
  )

  output = o_all.reshape(num_tokens, n_v * d_v)
  return new_recurrent_state, output
