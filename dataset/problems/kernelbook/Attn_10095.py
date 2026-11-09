"""[bf16 variant] Bahdanau concat attention: broadcast hidden, concat with encoder outputs, Linear, tanh, dot with v, softmax over time."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Attn_10095',
    'seq_len': 2048,
    'batch': 16,
    'hidden_size': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    T, B, H = CONFIG['seq_len'], CONFIG['batch'], CONFIG['hidden_size']
    k = jax.random.split(jax.random.key(0), 5)
    hidden = jax.random.uniform(k[0], (B, H), dtype)
    encoder_outputs = jax.random.uniform(k[1], (T, B, H), dtype)
    w = jax.random.normal(k[2], (2 * H, H), dtype) * 0.02
    b = jax.random.normal(k[3], (H,), dtype) * 0.02
    v = jax.random.normal(k[4], (H,), dtype) * 0.02
    return hidden, encoder_outputs, w, b, v


def workload(hidden, encoder_outputs, w, b, v):
    T, B, H = encoder_outputs.shape
    h = jnp.broadcast_to(hidden.reshape(B, 1, H), (B, T, H))
    enc = jnp.transpose(encoder_outputs, (1, 0, 2))
    cat = jnp.concatenate([h, enc], axis=2)
    energy = jnp.tanh(cat @ w + b)
    scores = jnp.einsum('bth,h->bt', energy, v)
    return jax.nn.softmax(scores, axis=1)[:, None, :]
