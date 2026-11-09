"""SequenceSummaryLayer: stack last-k hidden states at position 0 -> concat -> Linear -> Linear -> tanh."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SequenceSummaryLayer_1294',
    'num_layers': 12,
    'batch': 128,
    'seq_len': 64,
    'hidden_size': 1024,
    'summary_layers': 4,
}


def create_inputs(dtype=jnp.float32):
    L, B, S, H = CONFIG['num_layers'], CONFIG['batch'], CONFIG['seq_len'], CONFIG['hidden_size']
    k = CONFIG['summary_layers']
    keys = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(keys[0], (L, B, S, H), dtype)
    w1 = jax.random.normal(keys[1], (k * H, H), dtype) * 0.02
    b1 = jax.random.normal(keys[2], (H,), dtype) * 0.02
    w2 = jax.random.normal(keys[3], (H, H), dtype) * 0.02
    b2 = jax.random.normal(keys[4], (H,), dtype) * 0.02
    return x, w1, b1, w2, b2


def workload(x, w1, b1, w2, b2):
    H = x.shape[-1]
    k = w1.shape[0] // H
    B = x.shape[1]
    h = x[-k:, :, 0, :]  # (k, B, H)
    concat = jnp.transpose(h, (1, 0, 2)).reshape(B, k * H)
    resized = concat @ w1 + b1
    return jnp.tanh(resized @ w2 + b2)
