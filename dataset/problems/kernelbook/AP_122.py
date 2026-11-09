"""Linear -> AttentivePooling (Linear -> ReLU -> Linear -> +mask -> softmax -> weighted sum over axis 1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AP_122',
    'mask_batch': 8,
    'dim0': 8,
    'dim1': 16,
    'seq_len': 512,
    'input_dim': 512,
    'out_dim': 512,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    ks = jax.random.split(jax.random.key(0), 8)
    feature = jax.random.uniform(ks[0], (c['dim0'], c['dim1'], c['seq_len'], c['input_dim']), dtype)
    att_mask = jax.random.uniform(ks[1], (c['mask_batch'], c['dim0'], c['dim1'], c['seq_len']), dtype)
    w_lin = jax.random.normal(ks[2], (c['input_dim'], c['out_dim']), dtype) * 0.02
    b_lin = jax.random.normal(ks[3], (c['out_dim'],), dtype) * 0.02
    w_a = jax.random.normal(ks[4], (c['out_dim'], c['out_dim']), dtype) * 0.02
    b_a = jax.random.normal(ks[5], (c['out_dim'],), dtype) * 0.02
    w_w = jax.random.normal(ks[6], (c['out_dim'], 1), dtype) * 0.02
    b_w = jax.random.normal(ks[7], (1,), dtype) * 0.02
    return feature, att_mask, w_lin, b_lin, w_a, b_a, w_w, b_w


def workload(feature, att_mask, w_lin, b_lin, w_a, b_a, w_w, b_w):
    feat = feature @ w_lin + b_lin
    hidden = jax.nn.relu(feat @ w_a + b_a)
    att_logits = jnp.squeeze(hidden @ w_w + b_w, axis=-1)
    # Broadcasting matches torch: mask (M, D0, D1, T) + logits (D0, D1, T).
    att_logits = att_mask + att_logits
    att_w = jax.nn.softmax(att_logits, axis=-1)[..., None]
    # (D0, D1, T, O) * (M, D0, D1, T, 1) -> (M, D0, D1, T, O), summed over axis 1.
    return jnp.sum(feat * att_w, axis=1)
