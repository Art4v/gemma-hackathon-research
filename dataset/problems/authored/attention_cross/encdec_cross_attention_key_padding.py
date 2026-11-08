"""Encoder-decoder cross attention: Q/K/V projections -> per-example key-padding-masked softmax(QK^T/sqrt(d)) V -> output projection."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'encdec_cross_attention_key_padding',
    'batch': 16,
    'dec_len': 1024,
    'enc_len': 1536,
    'd_model': 1024,
    'num_heads': 16,
    'head_dim': 64,
    'min_enc_len_frac': 0.25,
    'mask_value': -1e30,
}


def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch']
    Sd = CONFIG['dec_len']
    Se = CONFIG['enc_len']
    D = CONFIG['d_model']
    HD = CONFIG['num_heads'] * CONFIG['head_dim']
    keys = jax.random.split(jax.random.key(0), 8)
    dec_x = jax.random.normal(keys[0], (B, Sd, D), dtype)
    enc_mem = jax.random.normal(keys[1], (B, Se, D), dtype)
    min_len = max(1, int(Se * CONFIG['min_enc_len_frac']))
    enc_lens = jax.random.randint(keys[2], (B,), min_len, Se + 1, dtype=jnp.int32)
    w_q = jax.random.normal(keys[3], (D, HD), dtype) * 0.02
    w_k = jax.random.normal(keys[4], (D, HD), dtype) * 0.02
    w_v = jax.random.normal(keys[5], (D, HD), dtype) * 0.02
    w_o = jax.random.normal(keys[6], (HD, D), dtype) * 0.02
    b_o = jnp.zeros((D,), dtype)
    return dec_x, enc_mem, enc_lens, w_q, w_k, w_v, w_o, b_o


def workload(dec_x, enc_mem, enc_lens, w_q, w_k, w_v, w_o, b_o):
    H = CONFIG['num_heads']
    dh = CONFIG['head_dim']
    B, Sd, D = dec_x.shape
    Se = enc_mem.shape[1]
    out_dtype = dec_x.dtype

    q = jnp.dot(dec_x, w_q, preferred_element_type=jnp.float32).astype(out_dtype)
    k = jnp.dot(enc_mem, w_k, preferred_element_type=jnp.float32).astype(out_dtype)
    v = jnp.dot(enc_mem, w_v, preferred_element_type=jnp.float32).astype(out_dtype)

    q = q.reshape(B, Sd, H, dh)
    k = k.reshape(B, Se, H, dh)
    v = v.reshape(B, Se, H, dh)

    scale = 1.0 / (dh ** 0.5)
    scores = jnp.einsum('bqhd,bkhd->bhqk', q, k,
                        preferred_element_type=jnp.float32) * scale

    key_valid = jnp.arange(Se, dtype=jnp.int32)[None, :] < enc_lens[:, None]  # [B, Se]
    scores = jnp.where(key_valid[:, None, None, :], scores,
                       jnp.float32(CONFIG['mask_value']))

    m = jnp.max(scores, axis=-1, keepdims=True)
    p = jnp.exp(scores - m)
    denom = jnp.sum(p, axis=-1, keepdims=True)
    probs = (p / denom).astype(out_dtype)

    ctx = jnp.einsum('bhqk,bkhd->bqhd', probs, v,
                     preferred_element_type=jnp.float32).astype(out_dtype)
    ctx = ctx.reshape(B, Sd, H * dh)

    out = jnp.dot(ctx, w_o, preferred_element_type=jnp.float32) + b_o.astype(jnp.float32)
    return out.astype(out_dtype)
