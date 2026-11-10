"""Additive self-attention pooling: tanh(x @ W1) @ W2 -> softmax over seq -> probs-weighted sum of x."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SelfAttention_8355',
    'batch': 32,
    'seq_len': 1024,
    'input_size': 1024,
    'hidden_size': 1024,
}


def create_inputs(dtype=jnp.float32):
    k_x, k_w1, k_w2 = jax.random.split(jax.random.key(0), 3)
    b, s, d, h = CONFIG['batch'], CONFIG['seq_len'], CONFIG['input_size'], CONFIG['hidden_size']
    x = jax.random.uniform(k_x, (b, s, d), dtype)
    w1 = jax.random.normal(k_w1, (d, h), dtype) * 0.02
    w2 = jax.random.normal(k_w2, (h, 1), dtype) * 0.02
    return x, w1, w2


def workload(x, w1, w2):
    b, s, d = x.shape
    scores = (jnp.tanh(x.reshape(-1, d) @ w1) @ w2).reshape(b, s)
    probs = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum('bs,bsd->bd', probs, x)
