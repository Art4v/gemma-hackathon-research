"""[bf16 variant] Linear(hidden->1) -> squeeze(-1) -> sigmoid -> multiply by float mask (Classifier_1798)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Classifier_1798',
    'batch': 64,
    'n_sents': 1024,
    'hidden_size': 1024,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_m, k_w, k_b = jax.random.split(jax.random.key(0), 4)
    b, s, h = CONFIG['batch'], CONFIG['n_sents'], CONFIG['hidden_size']
    x = jax.random.uniform(k_x, (b, s, h), dtype)
    mask_cls = jax.random.uniform(k_m, (b, s), dtype)
    w = jax.random.normal(k_w, (h, 1), dtype) * 0.02
    bias = jax.random.normal(k_b, (1,), dtype) * 0.02
    return x, mask_cls, w, bias


def workload(x, mask_cls, w, bias):
    h = jnp.squeeze(x @ w + bias, axis=-1)
    # Right-aligned broadcasting, same as torch, so mismatched mask ranks behave identically.
    return jax.nn.sigmoid(h) * mask_cls.astype(h.dtype)
