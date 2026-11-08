"""Encoder-decoder cross attention: Q/K/V projections -> key-padding-masked non-causal softmax(QK^T)V -> output projection."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'encdec_cross_attention_padmask',
    'batch': 8,
    'dec_len': 512,
    'enc_len': 3072,
    'd_model': 1280,
    'num_heads': 20,
    'head_dim': 64,
    'min_enc_len_frac': 4,  # minimum valid encoder length = enc_len // this
    'weight_scale': 0.02,
    'mask_value': -1e30,
}


def create_inputs(dtype=jnp.float32):
    B = CONFIG['batch']
    Tq = CONFIG['dec_len']
    Tk = CONFIG['enc_len']
    D = CONFIG['d_model']
    HD = CONFIG['num_heads'] * CONFIG['head_dim']
    s = CONFIG['weight_scale']
    key = jax.random.key(0)
    k_x, k_m, k_l, k_q, k_k, k_v, k_o = jax.random.split(key, 7)
    dec_x = jax.random.normal(k_x, (B, Tq, D), dtype)
    enc_mem = jax.random.normal(k_m, (B, Tk, D), dtype)
    enc_lens = jax.random.randint(
        k_l, (B,), Tk // CONFIG['min_enc_len_frac'], Tk + 1, dtype=jnp.int32)
    wq = jax.random.normal(k_q, (D, HD), dtype) * s
    wk = jax.random.normal(k_k, (D, HD), dtype) * s
    wv = jax.random.normal(k_v, (D, HD), dtype) * s
    wo = jax.random.normal(k_o, (HD, D), dtype) * s
    return dec_x, enc_mem, enc_lens, wq, wk, wv, wo


def workload(dec_x, enc_mem, enc_lens, wq, wk, wv, wo):
    B, Tq, D = dec_x.shape
    Tk = enc_mem.shape[1]
    H = CONFIG['num_heads']
    Dh = CONFIG['head_dim']
    out_dtype = dec_x.dtype

    q = jnp.einsum('btd,de->bte', dec_x, wq, preferred_element_type=jnp.float32)
    k = jnp.einsum('bsd,de->bse', enc_mem, wk, preferred_element_type=jnp.float32)
    v = jnp.einsum('bsd,de->bse', enc_mem, wv, preferred_element_type=jnp.float32)
    q = q.astype(out_dtype).reshape(B, Tq, H, Dh)
    k = k.astype(out_dtype).reshape(B, Tk, H, Dh)
    v = v.astype(out_dtype).reshape(B, Tk, H, Dh)

    scale = 1.0 / jnp.sqrt(jnp.float32(Dh))
    scores = jnp.einsum('bqhd,bkhd->bhqk', q, k,
                        preferred_element_type=jnp.float32) * scale
    valid = jnp.arange(Tk, dtype=jnp.int32)[None, :] < enc_lens[:, None]  # [B, Tk]
    scores = jnp.where(valid[:, None, None, :], scores,
                       jnp.float32(CONFIG['mask_value']))
    m = jnp.max(scores, axis=-1, keepdims=True)
    p = jnp.exp(scores - m)
    l = jnp.sum(p, axis=-1, keepdims=True)
    ctx = jnp.einsum('bhqk,bkhd->bqhd', p.astype(out_dtype), v,
                     preferred_element_type=jnp.float32)
    ctx = ctx / jnp.transpose(l, (0, 2, 1, 3))  # [B,Tq,H,1]
    ctx = ctx.astype(out_dtype).reshape(B, Tq, H * Dh)
    out = jnp.einsum('bte,ed->btd', ctx, wo, preferred_element_type=jnp.float32)
    return out.astype(out_dtype)
