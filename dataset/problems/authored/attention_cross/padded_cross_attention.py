"""Encoder-decoder cross attention: Q/K/V projections -> key-padding-masked softmax(QK^T/sqrt(d)) V -> output projection."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'padded_cross_attention',
    'batch': 16,
    'dec_len': 1024,
    'enc_len': 2048,
    'd_model': 1024,
    'num_heads': 16,
    'head_dim': 64,
    'min_enc_len_frac': 4,  # lengths drawn from [enc_len // 4, enc_len]
    'mask_value': -1e30,
}


def create_inputs(dtype=jnp.float32):
    B = CONFIG['batch']
    Tq = CONFIG['dec_len']
    Tk = CONFIG['enc_len']
    D = CONFIG['d_model']
    HD = CONFIG['num_heads'] * CONFIG['head_dim']
    key = jax.random.key(0)
    k_x, k_m, k_q, k_k, k_v, k_o, k_len = jax.random.split(key, 7)
    dec_hidden = jax.random.normal(k_x, (B, Tq, D), dtype)
    enc_memory = jax.random.normal(k_m, (B, Tk, D), dtype)
    wq = jax.random.normal(k_q, (D, HD), dtype) * 0.02
    wk = jax.random.normal(k_k, (D, HD), dtype) * 0.02
    wv = jax.random.normal(k_v, (D, HD), dtype) * 0.02
    wo = jax.random.normal(k_o, (HD, D), dtype) * 0.02
    enc_lengths = jax.random.randint(
        k_len, (B,), Tk // CONFIG['min_enc_len_frac'], Tk + 1, dtype=jnp.int32)
    return dec_hidden, enc_memory, enc_lengths, wq, wk, wv, wo


def workload(dec_hidden, enc_memory, enc_lengths, wq, wk, wv, wo):
    H = CONFIG['num_heads']
    Dh = CONFIG['head_dim']
    B, Tq, D = dec_hidden.shape
    Tk = enc_memory.shape[1]
    out_dtype = dec_hidden.dtype

    q = jnp.einsum('btd,de->bte', dec_hidden, wq,
                   preferred_element_type=jnp.float32).astype(out_dtype)
    k = jnp.einsum('bsd,de->bse', enc_memory, wk,
                   preferred_element_type=jnp.float32).astype(out_dtype)
    v = jnp.einsum('bsd,de->bse', enc_memory, wv,
                   preferred_element_type=jnp.float32).astype(out_dtype)

    q = q.reshape(B, Tq, H, Dh)
    k = k.reshape(B, Tk, H, Dh)
    v = v.reshape(B, Tk, H, Dh)

    scale = 1.0 / jnp.sqrt(jnp.float32(Dh))
    logits = jnp.einsum('bqhd,bkhd->bhqk', q, k,
                        preferred_element_type=jnp.float32) * scale

    valid = jnp.arange(Tk, dtype=jnp.int32)[None, :] < enc_lengths[:, None]  # [B, Tk]
    logits = jnp.where(valid[:, None, None, :], logits,
                       jnp.float32(CONFIG['mask_value']))

    m = jnp.max(logits, axis=-1, keepdims=True)
    p = jnp.exp(logits - m)
    p = jnp.where(valid[:, None, None, :], p, 0.0)
    denom = jnp.sum(p, axis=-1, keepdims=True)
    probs = p / jnp.maximum(denom, jnp.float32(1e-30))

    ctx = jnp.einsum('bhqk,bkhd->bqhd', probs.astype(out_dtype), v,
                     preferred_element_type=jnp.float32).astype(out_dtype)
    ctx = ctx.reshape(B, Tq, H * Dh)
    out = jnp.einsum('bte,ed->btd', ctx, wo,
                     preferred_element_type=jnp.float32)
    return out.astype(out_dtype)
