"""[bf16 variant] KeyValueAttention: linear key/query projection, tanh, linear score, max-shifted softmax, softmax-weighted value sum."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'KeyValueAttention_2258',
    'seq_len': 512,
    'batch': 128,
    'query_size': 256,
    'key_size': 256,
    'value_size': 256,
    'hid_size': 256,
    'output_size': 1,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    T = CONFIG['seq_len']
    B = CONFIG['batch']
    Q = CONFIG['query_size']
    K = CONFIG['key_size']
    V = CONFIG['value_size']
    H = CONFIG['hid_size']
    O = CONFIG['output_size']
    scale = CONFIG['weight_scale']

    keys = jax.random.split(jax.random.key(0), 6)
    q = jax.random.uniform(keys[0], (B, Q), dtype)
    k = jax.random.uniform(keys[1], (T, B, K), dtype)
    v = jax.random.uniform(keys[2], (T, B, V), dtype)
    key_w = jax.random.normal(keys[3], (K, H), dtype) * scale
    key_b = jnp.zeros((H,), dtype)
    query_w = jax.random.normal(keys[4], (Q, H), dtype) * scale
    query_b = jnp.zeros((H,), dtype)
    out_w = jax.random.normal(keys[5], (H, O), dtype) * scale
    out_b = jnp.zeros((O,), dtype)
    return q, k, v, key_w, key_b, query_w, query_b, out_w, out_b


def workload(q, k, v, key_w, key_b, query_w, query_b, out_w, out_b):
    k = jnp.swapaxes(k, 0, 1)
    v = jnp.swapaxes(v, 0, 1)
    h_k = k @ key_w + key_b
    h_q = q @ query_w + query_b
    h = jnp.tanh(h_k + h_q[:, None, :])
    logit = jnp.squeeze(h @ out_w + out_b, axis=2)
    logit = logit - jnp.max(logit, axis=1, keepdims=True)
    p = jax.nn.softmax(logit, axis=1)
    out = jnp.sum(v * p[:, :, None], axis=1, keepdims=True)
    out = jnp.swapaxes(out, 0, 1)
    return out, p
