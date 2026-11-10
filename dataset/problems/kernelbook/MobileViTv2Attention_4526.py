"""MobileViTv2 separable attention: linear -> softmax(axis=1) -> weighted linear -> sum(axis=1) -> linear * context -> linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MobileViTv2Attention_4526',
    'batch': 8,
    'n_tokens': 256,
    'n_groups': 256,
    'd_model': 128,
    'score_dim': 1,
    'attention_axis': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 9)
    d_model = CONFIG['d_model']
    score_dim = CONFIG['score_dim']
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(
        keys[0],
        (CONFIG['batch'], CONFIG['n_tokens'], CONFIG['n_groups'], d_model),
        dtype,
    )
    w_i = jax.random.normal(keys[1], (d_model, score_dim), dtype) * scale
    b_i = jax.random.normal(keys[2], (score_dim,), dtype) * scale
    w_k = jax.random.normal(keys[3], (d_model, d_model), dtype) * scale
    b_k = jax.random.normal(keys[4], (d_model,), dtype) * scale
    w_v = jax.random.normal(keys[5], (d_model, d_model), dtype) * scale
    b_v = jax.random.normal(keys[6], (d_model,), dtype) * scale
    w_o = jax.random.normal(keys[7], (d_model, d_model), dtype) * scale
    b_o = jax.random.normal(keys[8], (d_model,), dtype) * scale
    return x, w_i, b_i, w_k, b_k, w_v, b_v, w_o, b_o


def workload(x, w_i, b_i, w_k, b_k, w_v, b_v, w_o, b_o):
    axis = CONFIG['attention_axis']
    scores = x @ w_i + b_i
    weight_i = jax.nn.softmax(scores, axis=axis)
    context_score = weight_i * (x @ w_k + b_k)
    context_vector = jnp.sum(context_score, axis=axis, keepdims=True)
    v = (x @ w_v + b_v) * context_vector
    return v @ w_o + b_o
