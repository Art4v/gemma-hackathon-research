"""[bf16 variant] Copy_2209: tanh(Linear(enc)) einsum dec_hs -> [B, Tdec, Tenc] scaled by copy_weight."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Copy_2209',
    'batch': 16,
    'enc_len': 2048,
    'dec_len': 1024,
    'hidden_size': 1024,
    'copy_weight': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    k_enc, k_dec, k_w, k_b = jax.random.split(jax.random.key(0), 4)
    b, te, td, h = CONFIG['batch'], CONFIG['enc_len'], CONFIG['dec_len'], CONFIG['hidden_size']
    enc_out_hs = jax.random.uniform(k_enc, (b, te, h), dtype)
    dec_hs = jax.random.uniform(k_dec, (b, td, h), dtype)
    w = jax.random.normal(k_w, (h, h), dtype) * 0.02  # (in, out)
    bias = jax.random.normal(k_b, (h,), dtype) * 0.02
    return enc_out_hs, dec_hs, w, bias


def workload(enc_out_hs, dec_hs, w, bias):
    hp = jax.lax.Precision.HIGHEST
    proj = jnp.tanh(jnp.matmul(enc_out_hs, w, precision=hp) + bias)
    score = jnp.einsum('beh,bdh->bde', proj, dec_hs, precision=hp)
    return score * CONFIG['copy_weight']
