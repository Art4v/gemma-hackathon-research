import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp


CONFIGS = {
    'llama3_70b_quantized_splash_attention': {
        'name': 'llama3_70b_quantized_splash_attention',
        'model': 'Llama-3.1-70B-FP8',
        'operator': 'quantized_splash_attention',
        'batch': 4,
        'seq_len': 4096,
        'num_query_heads': 64,
        'num_kv_heads': 8,
        'head_dim': 128,
        'block_q': 128,
        'block_kv': 128,
        'atol': 1e-2,
        'rtol': 1e-2,
    },
    'llama3_8b_quantized_splash_attention': {
        'name': 'llama3_8b_quantized_splash_attention',
        'model': 'Llama-3.1-8B-FP8',
        'operator': 'quantized_splash_attention',
        'batch': 4,
        'seq_len': 4096,
        'num_query_heads': 32,
        'num_kv_heads': 8,
        'head_dim': 128,
        'block_q': 128,
        'block_kv': 128,
        'atol': 1e-2,
        'rtol': 1e-2,
    },
}


CONFIG = CONFIGS['llama3_70b_quantized_splash_attention']


def create_inputs(dtype=jnp.float8_e4m3fn, config=None):
  """Returns (q, k, v, q_scale, k_scale, v_scale)."""
  if config is None:
    cfg = CONFIG
  elif isinstance(config, str):
    cfg = CONFIGS[config]
  else:
    cfg = config
  key = jax.random.key(42)
  k1, k2, k3 = jax.random.split(key, 3)
  b = cfg['batch']
  s = cfg['seq_len']
  h_q = cfg['num_query_heads']
  h_kv = cfg['num_kv_heads']
  d = cfg['head_dim']

  q_f32 = jax.random.normal(k1, (b, h_q, s, d), dtype=jnp.float32)
  k_f32 = jax.random.normal(k2, (b, h_kv, s, d), dtype=jnp.float32)
  v_f32 = jax.random.normal(k3, (b, h_kv, s, d), dtype=jnp.float32)

  # Max-abs scale factors per token
  q_scale = (jnp.max(jnp.abs(q_f32), axis=-1, keepdims=True) / 448.0).astype(
      jnp.float32
  )
  k_scale = (jnp.max(jnp.abs(k_f32), axis=-1, keepdims=True) / 448.0).astype(
      jnp.float32
  )
  v_scale = (jnp.max(jnp.abs(v_f32), axis=-1, keepdims=True) / 448.0).astype(
      jnp.float32
  )

  q = (q_f32 / jnp.maximum(q_scale, 1e-6)).astype(dtype)
  k = (k_f32 / jnp.maximum(k_scale, 1e-6)).astype(dtype)
  v = (v_f32 / jnp.maximum(v_scale, 1e-6)).astype(dtype)
  return q, k, v, q_scale, k_scale, v_scale


import math
import jax
import jax.numpy as jnp


def workload(q, k, v, q_scale, k_scale, v_scale, block_q=None, block_kv=None):
  """Vectorized quantized causal attention in plain JAX."""
  del block_q, block_kv
  B, H_q, S, D = q.shape
  H_kv = k.shape[1]
  G = H_q // H_kv
  sm_scale = 1.0 / math.sqrt(D)

  # Dequantize to float32 and reshape Q for Grouped Query Attention (GQA)
  q_deq = (q.astype(jnp.float32) * q_scale).reshape(B, H_kv, G, S, D)
  k_deq = k.astype(jnp.float32) * k_scale
  v_deq = v.astype(jnp.float32) * v_scale

  # Causal mask: (S, S)
  idx = jnp.arange(S)
  mask = idx[:, None] >= idx[None, :]

  # Process per-(batch, kv_head) slice via vmap to avoid materializing
  # the full (B, H_q, S, S) attention matrix in HBM at once.
  @jax.vmap
  @jax.vmap
  def _attend_kv_head(q_g, k_h, v_h):
    # q_g: (G, S, D), k_h: (S, D), v_h: (S, D)
    logits = (
        jnp.einsum(
            'gsd,td->gst', q_g, k_h, preferred_element_type=jnp.float32
        )
        * sm_scale
    )
    logits = jnp.where(mask[None, :, :], logits, -1e30)
    m = jnp.max(logits, axis=-1, keepdims=True)
    p = jnp.exp(logits - m)
    l = jnp.sum(p, axis=-1, keepdims=True)
    acc = jnp.einsum('gst,td->gsd', p, v_h, preferred_element_type=jnp.float32)
    return (acc / jnp.maximum(l, 1e-6)).astype(jnp.bfloat16)

  out = _attend_kv_head(q_deq, k_deq, v_deq)
  return out.reshape(B, H_q, S, D)
