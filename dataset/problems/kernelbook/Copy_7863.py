"""[bf16 variant] Copy score: tanh(Linear(enc_out_hs)) einsum dec_hs -> [B, Tdec, Tenc], scaled by copy_weight."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Copy_7863',
    'batch': 16,
    't_enc': 2048,
    't_dec': 1024,
    'hidden_size': 1024,
    'copy_weight': 1.0,
}


def create_inputs(dtype=jnp.bfloat16):
    b, te, td, h = CONFIG['batch'], CONFIG['t_enc'], CONFIG['t_dec'], CONFIG['hidden_size']
    k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 4)
    enc_out_hs = jax.random.uniform(k1, (b, te, h), dtype)
    dec_hs = jax.random.uniform(k2, (b, td, h), dtype)
    w = jax.random.normal(k3, (h, h), dtype) * 0.02
    bias = jax.random.normal(k4, (h,), dtype) * 0.02
    return enc_out_hs, dec_hs, w, bias


def workload(enc_out_hs, dec_hs, w, bias):
    raw = jnp.tanh(enc_out_hs @ w + bias)
    score = jnp.einsum('beh,bdh->bde', raw, dec_hs)
    return score * CONFIG['copy_weight']
