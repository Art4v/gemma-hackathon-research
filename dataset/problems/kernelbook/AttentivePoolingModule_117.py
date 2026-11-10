"""Attentive pooling: Linear -> ReLU -> Linear -> squeeze -> add mask -> softmax -> weighted sum over axis 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AttentivePoolingModule_117',
    'mask_lead': 8,
    'batch': 8,
    'time': 128,
    'seq': 128,
    'hidden': 128,
    'score_dim': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 6)
    m = CONFIG['mask_lead']
    b = CONFIG['batch']
    t = CONFIG['time']
    s = CONFIG['seq']
    h = CONFIG['hidden']
    d = CONFIG['score_dim']
    scale = CONFIG['weight_scale']
    batch_rep = jax.random.uniform(keys[0], (b, t, s, h), dtype)
    att_mask = jax.random.uniform(keys[1], (m, b, t, s), dtype)
    w_a = jax.random.normal(keys[2], (h, h), dtype) * scale
    b_a = jax.random.normal(keys[3], (h,), dtype) * scale
    w = jax.random.normal(keys[4], (h, d), dtype) * scale
    b_w = jax.random.normal(keys[5], (d,), dtype) * scale
    return batch_rep, att_mask, w_a, b_a, w, b_w


def workload(batch_rep, att_mask, w_a, b_a, w, b_w):
    hidden = jax.nn.relu(batch_rep @ w_a + b_a)
    att_logits = jnp.squeeze(hidden @ w + b_w, axis=-1)
    att_logits = att_mask + att_logits
    att_w = jnp.expand_dims(jax.nn.softmax(att_logits, axis=-1), axis=-1)
    utter_rep = jnp.sum(batch_rep * att_w, axis=1)
    return utter_rep, att_w
