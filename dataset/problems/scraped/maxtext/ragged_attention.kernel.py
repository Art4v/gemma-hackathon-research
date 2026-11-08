"""Ragged grouped-query decode attention (Pallas TPU kernel, from MaxText)."""

import functools

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

BLOCK_SIZE = 256
DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


def ragged_flash_attention_kernel(
    lengths_ref,
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    m_ref,
    l_ref,
    *,
    block_size: int,
    mask_value: float,
):
  """Pallas kernel for flash attention."""
  b, i = pl.program_id(0), pl.program_id(1)

  @pl.when(i == 0)
  def init():
    m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
    l_ref[...] = jnp.zeros_like(l_ref)
    o_ref[...] = jnp.zeros_like(o_ref)

  length = lengths_ref[b]

  @pl.when(i * block_size < length)
  def run():
    q = q_ref[...].astype(jnp.float32)
    k = k_ref[...].astype(jnp.float32)
    v = v_ref[...].astype(jnp.float32)
    m_prev, l_prev = m_ref[...], l_ref[...]

    qk = lax.dot_general(q, k, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32)

    mask = i * block_size + jax.lax.broadcasted_iota(jnp.int32, qk.shape, 1) < length
    qk = qk + jnp.where(mask, 0.0, mask_value)
    m_curr = qk.max(axis=-1)

    s_curr = jnp.exp(qk - m_curr[..., None])
    l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
    o_curr_times_l_curr = jnp.dot(s_curr, v)

    m_curr = jax.lax.broadcast_in_dim(m_curr, m_prev.shape, (0,))
    m_next = jnp.maximum(m_prev, m_curr)
    alpha = jnp.exp(m_prev - m_next)
    beta = jnp.exp(m_curr - m_next)
    l_next = alpha * l_prev + beta * l_curr
    l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)

    m_ref[...], l_ref[...] = m_next, l_next_safe
    o_ref[...] = ((l_prev * alpha * o_ref[...] + beta * o_curr_times_l_curr) / l_next_safe).astype(o_ref.dtype)


def ragged_mqa(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    lengths: jax.Array,
    *,
    block_size: int = BLOCK_SIZE,
    mask_value: float = DEFAULT_MASK_VALUE,
):
  """Ragged multi query attention.

  q: [batch_size, num_heads, head_dim]; k, v: [batch_size, seq_len, head_dim];
  lengths: i32[batch_size].
  """
  batch_size, num_heads, head_dim = q.shape
  assert lengths.shape == (batch_size,)
  assert lengths.dtype == jnp.int32
  seq_len = k.shape[1]

  def compute_ragged_block_indices(b, i, lengths_ref):
    length = lengths_ref[b]
    not_done = i * block_size < length
    am_last_batch = b == batch_size - 1
    last_good_block = jnp.maximum(0, lax.div(length, block_size) - 1)
    b_next = jnp.where(not_done, b, jnp.where(am_last_batch, b, b + 1))
    i_next = jnp.where(not_done, i, jnp.where(am_last_batch, last_good_block, 0))
    return b_next, i_next, 0

  out, m, l = pl.pallas_call(
      functools.partial(
          ragged_flash_attention_kernel,
          block_size=block_size,
          mask_value=mask_value,
      ),
      grid_spec=pltpu.PrefetchScalarGridSpec(
          num_scalar_prefetch=1,
          in_specs=[
              pl.BlockSpec((None, num_heads, head_dim), lambda b, i, _: (b, 0, 0)),
              pl.BlockSpec((None, block_size, head_dim), compute_ragged_block_indices),
              pl.BlockSpec((None, block_size, head_dim), compute_ragged_block_indices),
          ],
          out_specs=[
              pl.BlockSpec((None, num_heads, head_dim), lambda b, i, _: (b, 0, 0)),
              pl.BlockSpec((None, num_heads, head_dim), lambda b, i, _: (b, 0, 0)),
              pl.BlockSpec((None, num_heads, head_dim), lambda b, i, _: (b, 0, 0)),
          ],
          grid=(batch_size, seq_len // block_size),
      ),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary")),
      out_shape=[
          jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), jnp.float32),
          jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), jnp.float32),
          jax.ShapeDtypeStruct((batch_size, num_heads, head_dim), jnp.float32),
      ],
  )(lengths, q, k, v)
  return out, m[..., 0], l[..., 0]


def workload(query, key, value, lengths):
  """Ragged GQA decode attention.

  query: [B, 1, Hq, D]; key, value: [B, S, Hkv, D]; lengths: i32[B].
  Returns normalized output [B, 1, Hq, D], max logits [B, 1, Hq, 1] and
  softmax denominators [B, 1, Hq, 1].
  """
  batch_size, _, num_heads_q, head_dim = query.shape
  _, _, num_heads_kv, _ = key.shape

  query = query.reshape(batch_size, num_heads_kv, num_heads_q // num_heads_kv, head_dim)
  key = jnp.swapaxes(key, 1, 2)
  value = jnp.swapaxes(value, 1, 2)

  o, m, l = jax.vmap(
      functools.partial(
          ragged_mqa,
          block_size=BLOCK_SIZE,
          mask_value=DEFAULT_MASK_VALUE,
      ),
      in_axes=(1, 1, 1, None),
      out_axes=1,
  )(query, key, value, lengths)

  m = jnp.reshape(m, (batch_size, 1, num_heads_q, 1))
  l = jnp.reshape(l, (batch_size, 1, num_heads_q, 1))
  o = jnp.reshape(o, (batch_size, 1, num_heads_q, head_dim))
  return o, m, l
