"""Concatenated attention: linear projections, broadcast, concat, ReLU, linear score, softmax over pixels, weighted sum."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'ConcatenatedAttention_3472',
    'batch': 32,
    'num_pixels': 1024,
    'encoder_dim': 1024,
    'decoder_dim': 1024,
    'attention_dim': 512,
    'score_dim': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    batch = CONFIG['batch']
    num_pixels = CONFIG['num_pixels']
    encoder_dim = CONFIG['encoder_dim']
    decoder_dim = CONFIG['decoder_dim']
    attention_dim = CONFIG['attention_dim']
    score_dim = CONFIG['score_dim']
    scale = CONFIG['weight_scale']

    keys = jax.random.split(jax.random.key(0), 8)
    encoder_out = jax.random.uniform(keys[0], (batch, num_pixels, encoder_dim), dtype)
    decoder_hidden = jax.random.uniform(keys[1], (batch, decoder_dim), dtype)
    w_enc = jax.random.normal(keys[2], (encoder_dim, attention_dim), dtype) * scale
    b_enc = jax.random.normal(keys[3], (attention_dim,), dtype) * scale
    w_dec = jax.random.normal(keys[4], (decoder_dim, attention_dim), dtype) * scale
    b_dec = jax.random.normal(keys[5], (attention_dim,), dtype) * scale
    w_full = jax.random.normal(keys[6], (attention_dim * 2, score_dim), dtype) * scale
    b_full = jax.random.normal(keys[7], (score_dim,), dtype) * scale
    return encoder_out, decoder_hidden, w_enc, b_enc, w_dec, b_dec, w_full, b_full


def workload(encoder_out, decoder_hidden, w_enc, b_enc, w_dec, b_dec, w_full, b_full):
    att1 = encoder_out @ w_enc + b_enc
    att2 = jnp.expand_dims(decoder_hidden @ w_dec + b_dec, axis=1)
    att2_expanded = jnp.broadcast_to(att2, att1.shape)
    concatenated = jnp.concatenate([att1, att2_expanded], axis=2)
    att = jnp.squeeze(jax.nn.relu(concatenated) @ w_full + b_full, axis=2)
    alpha = jax.nn.softmax(att, axis=1)
    attention_weighted_encoding = jnp.sum(encoder_out * jnp.expand_dims(alpha, axis=2), axis=1)
    return attention_weighted_encoding, alpha
