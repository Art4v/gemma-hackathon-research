"""AttentionUnit: linear x-embed + linear s-embed (expanded over T) -> tanh -> linear score -> softmax over T."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AttentionUnit_1455',
    'batch_size': 64,
    'T': 1024,
    'x_dim': 1024,
    's_dim': 1024,
    'att_dim': 1024,
    'score_dim': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 8)
    b = CONFIG['batch_size']
    t = CONFIG['T']
    x_dim = CONFIG['x_dim']
    s_dim = CONFIG['s_dim']
    att_dim = CONFIG['att_dim']
    score_dim = CONFIG['score_dim']
    scale = CONFIG['weight_scale']

    x = jax.random.uniform(keys[0], (b, t, x_dim), dtype)
    s_prev = jax.random.uniform(keys[1], (b, s_dim), dtype)
    s_w = jax.random.normal(keys[2], (s_dim, att_dim), dtype) * scale
    s_b = jax.random.normal(keys[3], (att_dim,), dtype) * scale
    x_w = jax.random.normal(keys[4], (x_dim, att_dim), dtype) * scale
    x_b = jax.random.normal(keys[5], (att_dim,), dtype) * scale
    w_w = jax.random.normal(keys[6], (att_dim, score_dim), dtype) * scale
    w_b = jax.random.normal(keys[7], (score_dim,), dtype) * scale
    return x, s_prev, s_w, s_b, x_w, x_b, w_w, w_b


def workload(x, s_prev, s_w, s_b, x_w, x_b, w_w, w_b):
    batch_size, t = x.shape[0], x.shape[1]
    x_proj = x @ x_w + x_b
    if s_prev.shape[0] == 1:
        s_prev = jnp.squeeze(s_prev, axis=0)
    s_proj = s_prev @ s_w + s_b
    s_proj = jnp.broadcast_to(s_proj[:, None, :], x_proj.shape)
    sum_tanh = jnp.tanh(s_proj + x_proj)
    v_proj = (sum_tanh @ w_w + w_b).reshape(batch_size, t)
    return jax.nn.softmax(v_proj, axis=1)
