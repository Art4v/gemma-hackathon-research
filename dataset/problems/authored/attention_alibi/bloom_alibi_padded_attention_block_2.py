"""BLOOM attention block: fused QKV projection -> padding-aware ALiBi (cumsum positions) + causal mask -> f32 softmax attention -> output projection."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'bloom_alibi_padded_attention_block',
    'batch': 2,
    'seq_len': 2048,
    'hidden': 4096,
    'num_heads': 32,
    'head_dim': 128,
    'min_len_frac': 0.5,
}


def _alibi_slopes(num_heads):
    # BLOOM / ALiBi slopes, handling non-power-of-2 head counts.
    import math
    closest = 2 ** math.floor(math.log2(num_heads))
    base = 2.0 ** (-(2.0 ** -(math.log2(closest) - 3)))
    slopes = [base ** (i + 1) for i in range(closest)]
    if closest != num_heads:
        extra_base = 2.0 ** (-(2.0 ** -(math.log2(2 * closest) - 3)))
        n_rem = min(closest, num_heads - closest)
        slopes += [extra_base ** (2 * i + 1) for i in range(n_rem)]
    return jnp.asarray(slopes, dtype=jnp.float32)


def create_inputs(dtype=jnp.float32):
    B, S, D = CONFIG['batch'], CONFIG['seq_len'], CONFIG['hidden']
    H, Dh = CONFIG['num_heads'], CONFIG['head_dim']
    key = jax.random.key(0)
    k_x, k_len, k_wqkv, k_bqkv, k_wo, k_bo = jax.random.split(key, 6)
    hidden_states = jax.random.normal(k_x, (B, S, D), dtype)
    min_len = int(S * CONFIG['min_len_frac'])
    lengths = jax.random.randint(k_len, (B,), min_len, S + 1, dtype=jnp.int32)
    w_qkv = jax.random.normal(k_wqkv, (D, 3 * H * Dh), dtype) * 0.02
    b_qkv = jax.random.normal(k_bqkv, (3 * H * Dh,), dtype) * 0.02
    w_o = jax.random.normal(k_wo, (H * Dh, D), dtype) * 0.02
    b_o = jax.random.normal(k_bo, (D,), dtype) * 0.02
    return hidden_states, lengths, w_qkv, b_qkv, w_o, b_o


def workload(hidden_states, lengths, w_qkv, b_qkv, w_o, b_o):
    B, S, D = hidden_states.shape
    H, Dh = CONFIG['num_heads'], CONFIG['head_dim']
    dtype = hidden_states.dtype

    # Fused QKV projection (f32 accumulate), BLOOM interleaved layout [H, 3, Dh].
    qkv = jnp.einsum('bsd,de->bse', hidden_states, w_qkv,
                     preferred_element_type=jnp.float32) + b_qkv.astype(jnp.float32)
    qkv = qkv.astype(dtype).reshape(B, S, H, 3, Dh)
    q = qkv[..., 0, :]
    k = qkv[..., 1, :]
    v = qkv[..., 2, :]

    # Left-padding attention mask and BLOOM cumsum-based ALiBi positions.
    pos = jnp.arange(S, dtype=jnp.int32)
    attn_mask = pos[None, :] >= (S - lengths)[:, None]  # [B, S] True = valid
    mask_i = attn_mask.astype(jnp.int32)
    key_pos = ((jnp.cumsum(mask_i, axis=-1) - 1) * mask_i).astype(jnp.float32)  # [B, S]
    slopes = _alibi_slopes(H)  # [H]
    alibi = slopes[None, :, None, None] * key_pos[:, None, None, :]  # [B, H, 1, S]

    scale = 1.0 / jnp.sqrt(jnp.float32(Dh))
    scores = jnp.einsum('bqhd,bkhd->bhqk', q, k,
                        preferred_element_type=jnp.float32) * scale
    scores = scores + alibi

    causal = pos[:, None] >= pos[None, :]  # [S, S]
    allowed = causal[None, None, :, :] & attn_mask[:, None, None, :]
    neg = jnp.finfo(jnp.float32).min
    scores = jnp.where(allowed, scores, neg)

    probs = jax.nn.softmax(scores, axis=-1)
    ctx = jnp.einsum('bhqk,bkhd->bqhd', probs.astype(dtype), v,
                     preferred_element_type=jnp.float32)
    ctx = ctx.astype(dtype).reshape(B, S, H * Dh)

    out = jnp.einsum('bse,ed->bsd', ctx, w_o,
                     preferred_element_type=jnp.float32) + b_o.astype(jnp.float32)
    return out.astype(dtype)
