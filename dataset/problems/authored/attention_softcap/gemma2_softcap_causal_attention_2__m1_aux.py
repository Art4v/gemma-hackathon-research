"""[m1 M2_second_output variant of authored/attention_softcap/gemma2_softcap_causal_attention_2] Gemma 2 soft-capped causal MHA that also returns the per-token absmax of the attention output over all heads (B, S), the scale for quantizing the output-projection input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'gemma2_softcap_causal_attention_m1_aux',
    'batch': 1,
    'num_heads': 16,
    'seq_len': 4096,
    'head_dim': 256,
    'query_pre_attn_scalar': 256.0,
    'attn_logit_softcap': 50.0,
}


def create_inputs(dtype=jnp.float32):
    B = CONFIG['batch']
    H = CONFIG['num_heads']
    S = CONFIG['seq_len']
    D = CONFIG['head_dim']
    key = jax.random.key(0)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (B, H, S, D), dtype=dtype)
    k = jax.random.normal(kk, (B, H, S, D), dtype=dtype)
    v = jax.random.normal(kv, (B, H, S, D), dtype=dtype)
    return q, k, v


def workload(q, k, v):
    S_q = q.shape[2]
    S_k = k.shape[2]
    scale = CONFIG['query_pre_attn_scalar'] ** -0.5
    cap = CONFIG['attn_logit_softcap']

    logits = jnp.einsum('bhqd,bhkd->bhqk', q, k,
                        preferred_element_type=jnp.float32)
    logits = logits * scale
    logits = cap * jnp.tanh(logits / cap)

    q_pos = jnp.arange(S_q)[:, None]
    k_pos = jnp.arange(S_k)[None, :]
    causal = k_pos <= q_pos
    logits = jnp.where(causal[None, None, :, :], logits,
                       jnp.finfo(jnp.float32).min)

    probs = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum('bhqk,bhkd->bhqd', probs.astype(v.dtype), v,
                     preferred_element_type=jnp.float32)
    out = out.astype(q.dtype)
    # Per-token absmax over every head's output, in f32.
    amax = jnp.max(jnp.abs(out.astype(jnp.float32)), axis=(1, 3))  # (B, S)
    return out, amax
