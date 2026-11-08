"""BLOOM attention block: fused QKV proj -> padding-aware ALiBi (non-pow2 heads) causal softmax attention -> output proj."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'bloom_alibi_padded_attention_block',
    'batch': 2,
    'seq_len': 2048,
    'num_heads': 40,
    'head_dim': 128,
    'hidden': 5120,
    'min_len': 1024,
    'weight_scale': 0.02,
}


def _bloom_alibi_slopes(num_heads):
    closest = 1 << (num_heads.bit_length() - 1)
    base = 2.0 ** (-8.0 / closest)
    slopes = base ** jnp.arange(1, closest + 1, dtype=jnp.float32)
    if closest != num_heads:
        extra_base = 2.0 ** (-8.0 / (2 * closest))
        n_rem = min(closest, num_heads - closest)
        extra = extra_base ** jnp.arange(1, 1 + 2 * n_rem, 2, dtype=jnp.float32)
        slopes = jnp.concatenate([slopes, extra], axis=0)
    return slopes


def create_inputs(dtype=jnp.float32):
    B, S, H = CONFIG['batch'], CONFIG['seq_len'], CONFIG['hidden']
    ws = CONFIG['weight_scale']
    k = jax.random.split(jax.random.key(0), 6)
    x = jax.random.normal(k[0], (B, S, H), dtype)
    lengths = jax.random.randint(k[1], (B,), CONFIG['min_len'], S + 1, dtype=jnp.int32)
    w_qkv = jax.random.normal(k[2], (H, 3 * H), dtype) * ws
    b_qkv = jax.random.normal(k[3], (3 * H,), dtype) * ws
    w_o = jax.random.normal(k[4], (H, H), dtype) * ws
    b_o = jax.random.normal(k[5], (H,), dtype) * ws
    return x, lengths, w_qkv, b_qkv, w_o, b_o


def workload(x, lengths, w_qkv, b_qkv, w_o, b_o):
    B, S, H = x.shape
    nh = CONFIG['num_heads']
    hd = H // nh
    dtype = x.dtype

    # Left-padded attention mask: valid tokens are the last `lengths[b]` positions.
    pos = jnp.arange(S, dtype=jnp.int32)
    attn_mask = (pos[None, :] >= (S - lengths)[:, None])  # [B, S]
    mask_i = attn_mask.astype(jnp.int32)

    # Fused QKV projection, BLOOM interleaved layout [B, S, nh, 3, hd].
    qkv = jnp.dot(x, w_qkv, preferred_element_type=jnp.float32) + b_qkv.astype(jnp.float32)
    qkv = qkv.astype(dtype).reshape(B, S, nh, 3, hd)
    q = qkv[:, :, :, 0, :]
    k = qkv[:, :, :, 1, :]
    v = qkv[:, :, :, 2, :]

    # Padding-aware ALiBi (key-position based, as in HF BLOOM).
    slopes = _bloom_alibi_slopes(nh)  # [nh]
    key_pos = ((jnp.cumsum(mask_i, axis=-1) - 1) * mask_i).astype(jnp.float32)  # [B, S]
    alibi = slopes[None, :, None, None] * key_pos[:, None, None, :]  # [B, nh, 1, S]

    scale = 1.0 / (hd ** 0.5)
    scores = jnp.einsum('bqhd,bkhd->bhqk', q, k, preferred_element_type=jnp.float32)
    scores = scores * scale + alibi

    causal = pos[:, None] >= pos[None, :]  # [S, S]
    allowed = causal[None, None, :, :] & attn_mask[:, None, None, :]
    neg = jnp.finfo(jnp.float32).min
    scores = jnp.where(allowed, scores, neg)
    probs = jax.nn.softmax(scores, axis=-1)
    probs = jnp.where(allowed, probs, 0.0)

    ctx = jnp.einsum('bhqk,bkhd->bqhd', probs.astype(dtype), v,
                     preferred_element_type=jnp.float32)
    ctx = ctx * attn_mask[:, :, None, None].astype(jnp.float32)  # zero padded query rows
    ctx = ctx.astype(dtype).reshape(B, S, H)

    out = jnp.dot(ctx, w_o, preferred_element_type=jnp.float32) + b_o.astype(jnp.float32)
    return out.astype(dtype)
