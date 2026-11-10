"""[bf16 variant] Attention_17406: Linear encoder/decoder projections, tanh, Linear score, softmax over axis 1, alpha-weighted sum."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Attention_17406',
    'encoder_batch': 8,
    'decoder_batch': 8,
    'spatial_1': 128,
    'spatial_2': 128,
    'encoder_dim': 128,
    'decoder_dim': 128,
    'attention_dim': 128,
    'score_dim': 1,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 8)
    encoder_shape = (CONFIG['encoder_batch'], CONFIG['spatial_1'], CONFIG['spatial_2'], CONFIG['encoder_dim'])
    decoder_shape = (CONFIG['decoder_batch'], CONFIG['spatial_1'], CONFIG['spatial_2'], CONFIG['decoder_dim'])
    encoder_out = jax.random.uniform(keys[0], encoder_shape, dtype)
    decoder_hidden = jax.random.uniform(keys[1], decoder_shape, dtype)
    w_enc = jax.random.normal(keys[2], (CONFIG['encoder_dim'], CONFIG['attention_dim']), dtype) * 0.02
    b_enc = jax.random.normal(keys[3], (CONFIG['attention_dim'],), dtype) * 0.02
    w_dec = jax.random.normal(keys[4], (CONFIG['decoder_dim'], CONFIG['attention_dim']), dtype) * 0.02
    b_dec = jax.random.normal(keys[5], (CONFIG['attention_dim'],), dtype) * 0.02
    w_full = jax.random.normal(keys[6], (CONFIG['attention_dim'], CONFIG['score_dim']), dtype) * 0.02
    b_full = jax.random.normal(keys[7], (CONFIG['score_dim'],), dtype) * 0.02
    return encoder_out, decoder_hidden, w_enc, b_enc, w_dec, b_dec, w_full, b_full


def workload(encoder_out, decoder_hidden, w_enc, b_enc, w_dec, b_dec, w_full, b_full):
    att1 = encoder_out @ w_enc + b_enc
    att2 = decoder_hidden @ w_dec + b_dec
    if att2.shape[0] == 1:
        att2 = jnp.squeeze(att2, axis=0)
    att = jnp.tanh(att1 + jnp.expand_dims(att2, axis=1)) @ w_full + b_full
    if att.shape[2] == 1:
        att = jnp.squeeze(att, axis=2)
    alpha = jax.nn.softmax(att, axis=1)
    attention_weighted_encoding = jnp.sum(att1 * jnp.expand_dims(alpha, axis=2), axis=1)
    return attention_weighted_encoding, alpha
