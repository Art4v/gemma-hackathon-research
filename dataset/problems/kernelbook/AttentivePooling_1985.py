"""[bf16 variant] AttentivePooling: Linear -> ReLU -> Linear(1) -> +mask -> softmax -> weighted sum over axis 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AttentivePooling_1985',
    'batch': 32,
    'seq_len': 2048,
    'hidden': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    b, t, h = CONFIG['batch'], CONFIG['seq_len'], CONFIG['hidden']
    k = jax.random.split(jax.random.key(0), 6)
    batch_rep = jax.random.normal(k[0], (b, t, h), dtype)
    att_mask = jax.random.uniform(k[1], (b, t), dtype)
    w_a = jax.random.normal(k[2], (h, h), dtype) * 0.02
    b_a = jax.random.normal(k[3], (h,), dtype) * 0.02
    w = jax.random.normal(k[4], (h, 1), dtype) * 0.02
    b_w = jax.random.normal(k[5], (1,), dtype) * 0.02
    return batch_rep, att_mask, w_a, b_a, w, b_w


def workload(batch_rep, att_mask, w_a, b_a, w, b_w):
    # Rank-agnostic, mirrors torch broadcasting (squeeze(-1), mask + logits, unsqueeze(-1)).
    hidden = jax.nn.relu(batch_rep @ w_a + b_a)
    att_logits = (hidden @ w + b_w)[..., 0]
    att_logits = att_mask + att_logits
    att_w = jax.nn.softmax(att_logits, axis=-1)[..., None]
    utter_rep = jnp.sum(batch_rep * att_w, axis=1)
    return utter_rep, att_w
