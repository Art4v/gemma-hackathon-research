"""[bf16 variant] AttentionUnit: linear projections of encoder states and decoder state, sum, tanh, linear score, softmax over time."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AttentionUnit_1951',
    'batch': 128,
    'time_steps': 512,
    'x_dim': 512,
    's_dim': 512,
    'att_dim': 512,
    'score_dim': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    keys = jax.random.split(jax.random.key(0), 8)
    batch = CONFIG['batch']
    time_steps = CONFIG['time_steps']
    x_dim = CONFIG['x_dim']
    s_dim = CONFIG['s_dim']
    att_dim = CONFIG['att_dim']
    score_dim = CONFIG['score_dim']
    scale = CONFIG['weight_scale']

    x = jax.random.uniform(keys[0], (batch, time_steps, x_dim), dtype)
    s_prev = jax.random.uniform(keys[1], (batch, s_dim), dtype)
    s_weight = jax.random.normal(keys[2], (s_dim, att_dim), dtype) * scale
    s_bias = jax.random.normal(keys[3], (att_dim,), dtype) * scale
    x_weight = jax.random.normal(keys[4], (x_dim, att_dim), dtype) * scale
    x_bias = jax.random.normal(keys[5], (att_dim,), dtype) * scale
    w_weight = jax.random.normal(keys[6], (att_dim, score_dim), dtype) * scale
    w_bias = jax.random.normal(keys[7], (score_dim,), dtype) * scale
    return x, s_prev, s_weight, s_bias, x_weight, x_bias, w_weight, w_bias


def workload(x, s_prev, s_weight, s_bias, x_weight, x_bias, w_weight, w_bias):
    batch, time_steps = x.shape[0], x.shape[1]
    att_dim = s_weight.shape[1]

    x_proj = x @ x_weight + x_bias

    if s_prev.shape[0] == 1:
        s_prev = jnp.squeeze(s_prev, 0)
    s_proj = s_prev @ s_weight + s_bias
    s_proj = jnp.expand_dims(s_proj, 1)
    s_proj = jnp.broadcast_to(s_proj, (batch, time_steps, att_dim))

    sum_tanh = jnp.tanh(s_proj + x_proj)
    v_proj = sum_tanh @ w_weight + w_bias
    v_proj = v_proj.reshape(batch, time_steps)
    return jax.nn.softmax(v_proj, axis=1)
