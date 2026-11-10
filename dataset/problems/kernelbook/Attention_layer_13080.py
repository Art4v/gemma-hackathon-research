"""Attention_layer: permute -> linear (with bias) -> softmax(dim=-1) -> permute -> elementwise multiply with input."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Attention_layer_13080',
    'batch': 16,
    'sequence_length': 1024,
    'feature_length': 1024,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    k_x, k_w, k_b = jax.random.split(key, 3)
    seq = CONFIG['sequence_length']
    x = jax.random.uniform(
        k_x, (CONFIG['batch'], seq, CONFIG['feature_length']), dtype
    )
    w = jax.random.normal(k_w, (seq, seq), dtype) * CONFIG['weight_scale']
    b = jax.random.normal(k_b, (seq,), dtype) * CONFIG['weight_scale']
    return x, w, b


def workload(x, w, b):
    p = jnp.transpose(x, (0, 2, 1))
    y = jax.nn.softmax(p @ w + b, axis=-1)
    y = jnp.transpose(y, (0, 2, 1))
    return x * y
