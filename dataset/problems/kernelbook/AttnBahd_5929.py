"""Bahdanau attention: linear(enc) + linear(dec) -> tanh -> linear score -> reshape (B, SL) -> log_softmax."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AttnBahd_5929',
    'seq_len': 512,
    'batch': 128,
    'enc_dim': 640,
    'dec_dim': 384,
    'attn_dim': 256,
    'score_dim': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    sl = CONFIG['seq_len']
    b = CONFIG['batch']
    h = CONFIG['enc_dim']
    s = CONFIG['dec_dim']
    a = CONFIG['attn_dim']
    o = CONFIG['score_dim']
    scale = CONFIG['weight_scale']
    keys = jax.random.split(jax.random.key(0), 8)
    prev_h = jax.random.uniform(keys[0], (b, s), dtype)
    enc_outputs = jax.random.uniform(keys[1], (sl, b, h), dtype)
    u_w = jax.random.normal(keys[2], (h, a), dtype) * scale
    u_b = jax.random.normal(keys[3], (a,), dtype) * scale
    w_w = jax.random.normal(keys[4], (s, a), dtype) * scale
    w_b = jax.random.normal(keys[5], (a,), dtype) * scale
    v_w = jax.random.normal(keys[6], (a, o), dtype) * scale
    v_b = jax.random.normal(keys[7], (o,), dtype) * scale
    return prev_h, enc_outputs, u_w, u_b, w_w, w_b, v_w, v_b


def workload(prev_h, enc_outputs, u_w, u_b, w_w, w_b, v_w, v_b):
    src_seq_len, batch_size, _ = enc_outputs.shape
    uh = enc_outputs @ u_w + u_b
    wq = (prev_h.reshape(-1, w_w.shape[0]) @ w_w + w_b)[None, ...]
    wquh = jnp.tanh(wq + uh)
    scores = (wquh @ v_w + v_b).reshape(batch_size, src_seq_len)
    return jax.nn.log_softmax(scores, axis=1)
