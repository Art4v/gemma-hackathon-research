"""Luong attention: concat(broadcast hidden, enc) -> linear -> tanh -> v-score -> softmax * mask -> renorm -> bmm context."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Attention_395',
    'batch': 32,
    'seq_len': 1024,
    'hidden_size': 1024,
    'hidden_size1': 1024,
    'epsilon': 1e-10,
}


def create_inputs(dtype=jnp.float32):
    b, s = CONFIG['batch'], CONFIG['seq_len']
    hd, h1 = CONFIG['hidden_size'], CONFIG['hidden_size1']
    k1, k2, k3, k4, k5 = jax.random.split(jax.random.key(0), 5)
    encoder_outputs = jax.random.uniform(k1, (b, s, h1), dtype)
    decoder_hidden = jax.random.uniform(k2, (b, hd), dtype)
    inp_mask = jax.random.uniform(k3, (b, s), dtype)
    w_h = jax.random.normal(k4, (hd + h1, hd), dtype) * 0.02
    v = jax.random.normal(k5, (hd,), dtype) * 0.02
    return encoder_outputs, decoder_hidden, inp_mask, w_h, v


def workload(encoder_outputs, decoder_hidden, inp_mask, w_h, v):
    b, s, _ = encoder_outputs.shape
    hd = decoder_hidden.shape[-1]
    h = jnp.broadcast_to(decoder_hidden[:, None, :], (b, s, hd))
    energy = jnp.tanh(jnp.concatenate([h, encoder_outputs], axis=2) @ w_h)
    scores = energy @ v
    a = jax.nn.softmax(scores, axis=-1) * inp_mask
    a = a / (a.sum(axis=1, keepdims=True) + CONFIG['epsilon'])
    a = a[:, None, :]
    context = a @ encoder_outputs
    return a, context
