"""Ragged grouped-query decode attention (reference).

Single-token decode attention over a padded KV cache where each batch entry has
its own valid length. Mirrors MaxText's `ragged_gqa` Pallas kernel: query heads
are grouped onto KV heads, logits are computed in float32, positions at or
beyond `lengths[b]` are masked with a large negative value, and the kernel
returns the normalized attention output together with the running max logit
and softmax denominator.
"""

import jax
import jax.numpy as jnp
import numpy as np

CONFIG = {
    "batch_size": 8,
    "num_q_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "seq_len": 2048,
    "block_size": 256,
}

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


def create_inputs(dtype=jnp.bfloat16):
  key = jax.random.key(0)
  kq, kk, kv, kl = jax.random.split(key, 4)
  b = CONFIG["batch_size"]
  hq = CONFIG["num_q_heads"]
  hkv = CONFIG["num_kv_heads"]
  d = CONFIG["head_dim"]
  s = CONFIG["seq_len"]
  query = jax.random.normal(kq, (b, 1, hq, d), dtype=jnp.float32).astype(dtype)
  key_cache = jax.random.normal(kk, (b, s, hkv, d), dtype=jnp.float32).astype(dtype)
  value_cache = jax.random.normal(kv, (b, s, hkv, d), dtype=jnp.float32).astype(dtype)
  lengths = jax.random.randint(kl, (b,), 1, s + 1, dtype=jnp.int32)
  return query, key_cache, value_cache, lengths


def workload(query, key, value, lengths):
  batch_size, _, num_heads_q, head_dim = query.shape
  _, seq_len, num_heads_kv, _ = key.shape
  group = num_heads_q // num_heads_kv

  q = query.reshape(batch_size, num_heads_kv, group, head_dim).astype(jnp.float32)
  k = jnp.swapaxes(key, 1, 2).astype(jnp.float32)
  v = jnp.swapaxes(value, 1, 2).astype(jnp.float32)

  logits = jnp.einsum("bhgd,bhtd->bhgt", q, k)
  mask = jnp.arange(seq_len)[None, :] < lengths[:, None]
  logits = logits + jnp.where(mask, 0.0, DEFAULT_MASK_VALUE)[:, None, None, :]
  m = logits.max(axis=-1)
  p = jnp.exp(logits - m[..., None])
  l = p.sum(axis=-1)
  o = jnp.einsum("bhgt,bhtd->bhgd", p, v) / l[..., None]

  o = o.reshape(batch_size, 1, num_heads_q, head_dim)
  m = m.reshape(batch_size, 1, num_heads_q, 1)
  l = l.reshape(batch_size, 1, num_heads_q, 1)
  return o, m, l
