"""[m1 M2_second_output variant of authored/attention_alibi/alibi_causal_mha] Causal ALiBi multi-head attention that also returns the per-row softmax logsumexp (B, H, S) in f32."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'alibi_causal_mha_m1_aux',
    'batch': 2,
    'num_heads': 32,
    'seq_len': 2048,
    'head_dim': 128,
    'alibi_max_bias': 8.0,  # slopes are 2^(-alibi_max_bias * (h+1) / num_heads)
}


def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch']
    H = CONFIG['num_heads']
    S = CONFIG['seq_len']
    D = CONFIG['head_dim']
    kq, kk, kv = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(kq, (B, H, S, D), dtype)
    k = jax.random.normal(kk, (B, H, S, D), dtype)
    v = jax.random.normal(kv, (B, H, S, D), dtype)
    # ALiBi slopes: deterministic geometric sequence per head (Press et al., 2022).
    heads = jnp.arange(1, H + 1, dtype=jnp.float32)
    slopes = jnp.exp2(-CONFIG['alibi_max_bias'] * heads / H).astype(jnp.float32)
    return q, k, v, slopes


def workload(q, k, v, slopes):
    B, H, S, D = q.shape
    scale = 1.0 / (D ** 0.5)

    # QK^T in f32 accumulation.
    scores = jnp.einsum('bhqd,bhkd->bhqk', q, k,
                        preferred_element_type=jnp.float32) * scale

    # ALiBi bias: -slope_h * (i - j) for j <= i; causal mask for j > i.
    qpos = jnp.arange(S, dtype=jnp.int32)[:, None]
    kpos = jnp.arange(S, dtype=jnp.int32)[None, :]
    rel = (kpos - qpos).astype(jnp.float32)  # <= 0 on and below diagonal
    bias = slopes[:, None, None] * rel[None, :, :]  # (H, S, S), non-positive
    causal = kpos <= qpos  # (S, S)
    neg = jnp.asarray(jnp.finfo(jnp.float32).min, jnp.float32)
    scores = scores + bias[None]
    scores = jnp.where(causal[None, None], scores, neg)

    # Softmax in f32.
    m = jnp.max(scores, axis=-1, keepdims=True)
    p = jnp.exp(scores - m)
    l = jnp.sum(p, axis=-1, keepdims=True)
    p = p / l
    # Row logsumexp of the biased, masked scores (what a flash kernel saves for backward).
    lse = (m + jnp.log(l))[..., 0]  # (B, H, S) f32

    # PV with bf16 probabilities, f32 accumulation.
    out = jnp.einsum('bhqk,bhkd->bhqd', p.astype(v.dtype), v,
                     preferred_element_type=jnp.float32)
    return out.astype(q.dtype), lse
