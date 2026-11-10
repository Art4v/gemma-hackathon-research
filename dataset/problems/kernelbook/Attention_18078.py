"""[bf16 variant] Attention_18078: repeat + concat + linear (no bias) + tanh + dot with V + broadcast add of attention mask + softmax over axis 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Attention_18078',
    'batch': 128,
    'seq_len': 128,
    'n_hidden_enc': 512,
    'n_hidden_dec': 512,
    'mask_outer': 16,
    'mask_inner': 128,
    'softmax_axis': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    b = CONFIG['batch']
    s = CONFIG['seq_len']
    enc = CONFIG['n_hidden_enc']
    dec = CONFIG['n_hidden_dec']
    k_hidden, k_enc, k_mask, k_w, k_v = jax.random.split(jax.random.key(0), 5)
    hidden_dec = jax.random.uniform(k_hidden, (b, dec), dtype)
    last_layer_enc = jax.random.uniform(k_enc, (b, s, enc), dtype)
    attention_mask = jax.random.uniform(
        k_mask, (CONFIG['mask_outer'], CONFIG['mask_inner'], b, s), dtype
    )
    w = jax.random.normal(k_w, (enc + dec, dec), dtype) * 0.02
    v = jax.random.normal(k_v, (dec,), dtype) * 0.02
    return hidden_dec, last_layer_enc, attention_mask, w, v


def workload(hidden_dec, last_layer_enc, attention_mask, w, v):
    seq_len = last_layer_enc.shape[1]
    repeated = jnp.repeat(hidden_dec[:, None, :], seq_len, axis=1)
    combined = jnp.concatenate((repeated, last_layer_enc), axis=-1)
    tanh_w_s_h = jnp.tanh(combined @ w)
    e = tanh_w_s_h @ v
    e = attention_mask + e
    return jax.nn.softmax(e, axis=CONFIG['softmax_axis'])
