"""Luong general attention: Linear(encoder_outputs) -> dot with hidden -> softmax over seq."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Attn_15410',
    'batch': 32,
    'seq_len': 2048,
    'hidden_size': 1024,
}


def create_inputs(dtype=jnp.float32):
    b, s, h = CONFIG['batch'], CONFIG['seq_len'], CONFIG['hidden_size']
    k_h, k_e, k_w, k_b = jax.random.split(jax.random.key(0), 4)
    hidden = jax.random.uniform(k_h, (b, h), dtype)
    encoder_outputs = jax.random.uniform(k_e, (b, s, h), dtype)
    w = jax.random.normal(k_w, (h, h), dtype) * 0.02
    bias = jax.random.normal(k_b, (h,), dtype) * 0.02
    return hidden, encoder_outputs, w, bias


def workload(hidden, encoder_outputs, w, bias):
    energy = jnp.einsum('bsh,ho->bso', encoder_outputs, w) + bias
    scores = jnp.einsum('bso,bo->bs', energy, hidden)
    return jax.nn.softmax(scores, axis=1)[:, None, :]
