"""SelfAttentionPooling: Linear(H->1) logits + additive mask -> softmax over T -> weighted sum over T."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SelfAttentionPooling_17848',
    'batch': 16,
    'seq_len': 2048,
    'hidden': 1024,
}


def create_inputs(dtype=jnp.float32):
    n, t, h = CONFIG['batch'], CONFIG['seq_len'], CONFIG['hidden']
    k_x, k_m, k_w = jax.random.split(jax.random.key(0), 3)
    batch_rep = jax.random.uniform(k_x, (n, t, h), dtype)
    att_mask = jax.random.uniform(k_m, (n, t), dtype)
    w = jax.random.normal(k_w, (h, 1), dtype) * 0.02
    b = jnp.zeros((1,), dtype)
    return batch_rep, att_mask, w, b


def workload(batch_rep, att_mask, w, b):
    # Generic broadcasting mirrors torch exactly, including KernelBench's 4-D test inputs.
    att_logits = (batch_rep @ w + b)[..., 0]
    att_logits = att_mask + att_logits
    att_w = jax.nn.softmax(att_logits, axis=-1)[..., None]
    return jnp.sum(batch_rep * att_w, axis=1)
