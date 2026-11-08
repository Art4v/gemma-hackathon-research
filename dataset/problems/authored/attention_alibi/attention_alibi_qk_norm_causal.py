"""Multi-head causal attention with QK-RMSNorm and ALiBi positional bias."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'attention_alibi_qk_norm_causal',
    'batch_size': 4,
    'num_heads': 32,
    'seq_len': 2048,
    'head_dim': 128,
    'scale': 0.08838834764831845,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    B = CONFIG['batch_size']
    H = CONFIG['num_heads']
    S = CONFIG['seq_len']
    D = CONFIG['head_dim']
    q = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
    k = jax.random.normal(k2, (B, H, S, D), dtype=dtype)
    v = jax.random.normal(k3, (B, H, S, D), dtype=dtype)
    q_weight = jnp.ones((H, D), dtype=dtype)
    k_weight = jnp.ones((H, D), dtype=dtype)
    head_indices = jnp.arange(1, H + 1, dtype=jnp.float32)
    slopes = jnp.power(2.0, -8.0 * head_indices / H)
    return q, k, v, q_weight, k_weight, slopes


def workload(q, k, v, q_weight, k_weight, slopes):
    S = CONFIG['seq_len']
    scale = CONFIG['scale']
    eps = CONFIG['eps']

    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)

    q_rms = jnp.sqrt(jnp.mean(jnp.square(q_f32), axis=-1, keepdims=True) + eps)
    k_rms = jnp.sqrt(jnp.mean(jnp.square(k_f32), axis=-1, keepdims=True) + eps)

    q_norm = (q_f32 / q_rms) * q_weight[:, None, :].astype(jnp.float32)
    k_norm = (k_f32 / k_rms) * k_weight[:, None, :].astype(jnp.float32)

    q_norm = q_norm.astype(q.dtype)
    k_norm = k_norm.astype(k.dtype)

    scores = jnp.einsum('bhsd,bhtd->bhst', q_norm, k_norm, preferred_element_type=jnp.float32)
    scores = scores * scale

    pos = jnp.arange(S, dtype=jnp.float32)
    rel_pos = pos[None, :] - pos[:, None]
    alibi_bias = slopes[:, None, None] * rel_pos[None, :, :]
    scores = scores + alibi_bias[None, :, :, :]

    causal_mask = rel_pos <= 0.0
    scores = jnp.where(causal_mask[None, None, :, :], scores, -1e30)

    scores_max = jnp.max(scores, axis=-1, keepdims=True)
    exp_scores = jnp.exp(scores - scores_max)
    attn_weights = exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)

    out = jnp.einsum('bhst,bhtd->bhsd', attn_weights.astype(v.dtype), v, preferred_element_type=jnp.float32)
    return out.astype(q.dtype)
