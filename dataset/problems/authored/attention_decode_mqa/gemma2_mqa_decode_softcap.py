"""Single-token decode multi-query attention with query RMSNorm, logit soft-capping, and per-sequence valid cache lengths."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gemma2_mqa_decode_softcap',
    'batch_size': 64,
    'num_q_heads': 32,
    'max_kv_len': 16384,
    'head_dim': 128,
    'softcap': 50.0,
    'eps': 1e-6,
    'min_valid_len': 1024,
    'mask_value': -1e30,
}

def create_inputs(dtype=jnp.bfloat16):
    k0, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    B = CONFIG['batch_size']
    H = CONFIG['num_q_heads']
    S = CONFIG['max_kv_len']
    D = CONFIG['head_dim']

    q = jax.random.normal(k0, (B, H, D), dtype=dtype)
    k_cache = jax.random.normal(k1, (B, S, D), dtype=dtype)
    v_cache = jax.random.normal(k2, (B, S, D), dtype=dtype)
    valid_lens = jax.random.randint(k3, (B,), CONFIG['min_valid_len'], S + 1, dtype=jnp.int32)
    q_norm_scale = jnp.ones((D,), dtype=dtype)

    return q, k_cache, v_cache, valid_lens, q_norm_scale

def workload(q, k_cache, v_cache, valid_lens, q_norm_scale):
    orig_dtype = q.dtype
    D = CONFIG['head_dim']
    S = CONFIG['max_kv_len']
    softcap = CONFIG['softcap']
    eps = CONFIG['eps']
    mask_value = CONFIG['mask_value']

    q_f32 = q.astype(jnp.float32)
    k_f32 = k_cache.astype(jnp.float32)
    v_f32 = v_cache.astype(jnp.float32)
    scale_f32 = q_norm_scale.astype(jnp.float32)

    variance = jnp.mean(jnp.square(q_f32), axis=-1, keepdims=True)
    q_normed = q_f32 * jax.lax.rsqrt(variance + eps) * scale_f32
    q_scaled = q_normed * (D ** -0.5)

    logits = jnp.einsum('bhd,bsd->bhs', q_scaled, k_f32)
    logits = jnp.tanh(logits / softcap) * softcap

    pos = jnp.arange(S, dtype=jnp.int32)[None, None, :]
    mask = pos < valid_lens[:, None, None]
    logits = jnp.where(mask, logits, mask_value)

    weights = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum('bhs,bsd->bhd', weights, v_f32)

    return out.astype(orig_dtype)
