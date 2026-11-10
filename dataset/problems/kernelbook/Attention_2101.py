"""[bf16 variant] Attention_2101: affine(encoder) + affine(decoder) -> ReLU -> affine(1) -> softmax -> attention-weighted sum."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Attention_2101',
    'batch_size': 32,
    'num_pixels': 1024,
    'encoder_dim': 1024,
    'decoder_dim': 512,
    'attention_dim': 512,
    'full_att_out': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 8)
    b = CONFIG['batch_size']
    n = CONFIG['num_pixels']
    e = CONFIG['encoder_dim']
    d = CONFIG['decoder_dim']
    a = CONFIG['attention_dim']
    o = CONFIG['full_att_out']
    s = CONFIG['weight_scale']
    encoder_out = jax.random.uniform(keys[0], (b, n, e), dtype)
    decoder_hidden = jax.random.uniform(keys[1], (b, d), dtype)
    w_encoder = jax.random.normal(keys[2], (e, a), dtype) * s
    b_encoder = jax.random.normal(keys[3], (a,), dtype) * s
    w_decoder = jax.random.normal(keys[4], (d, a), dtype) * s
    b_decoder = jax.random.normal(keys[5], (a,), dtype) * s
    w_full = jax.random.normal(keys[6], (a, o), dtype) * s
    b_full = jax.random.normal(keys[7], (o,), dtype) * s
    return (
        encoder_out,
        decoder_hidden,
        w_encoder,
        b_encoder,
        w_decoder,
        b_decoder,
        w_full,
        b_full,
    )


def workload(
    encoder_out,
    decoder_hidden,
    w_encoder,
    b_encoder,
    w_decoder,
    b_decoder,
    w_full,
    b_full,
):
    att1 = encoder_out @ w_encoder + b_encoder
    att2 = decoder_hidden @ w_decoder + b_decoder
    hidden = jax.nn.relu(att1 + jnp.expand_dims(att2, 1))
    att = hidden @ w_full + b_full
    if att.shape[2] == 1:
        att = jnp.squeeze(att, 2)
    alpha = jax.nn.softmax(att, axis=1)
    attention_weighted_encoding = jnp.sum(
        encoder_out * jnp.expand_dims(alpha, 2), axis=1
    )
    return attention_weighted_encoding, alpha
