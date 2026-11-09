"""[bf16 variant] AttentionPooling: tanh(Linear_u) -> Linear_t score -> masked softmax over seq -> weighted sum of uq -> tanh(Linear_o)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AttentionPooling_4576',
    'seq_len': 1024,
    'batch': 64,
    'input_size': 1024,
    'output_size': 1024,
    'softmax_eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    s, b = CONFIG['seq_len'], CONFIG['batch']
    i, o = CONFIG['input_size'], CONFIG['output_size']
    ks = jax.random.split(jax.random.key(0), 8)
    uq = jax.random.uniform(ks[0], (s, b, i), dtype)
    mask = jax.random.uniform(ks[1], (b, s), dtype)
    w_u = jax.random.normal(ks[2], (i, o), dtype) * 0.02
    b_u = jax.random.normal(ks[3], (o,), dtype) * 0.02
    w_t = jax.random.normal(ks[4], (o, 1), dtype) * 0.02
    b_t = jax.random.normal(ks[5], (1,), dtype) * 0.02
    w_o = jax.random.normal(ks[6], (i, o), dtype) * 0.02
    b_o = jax.random.normal(ks[7], (o,), dtype) * 0.02
    return uq, mask, w_u, b_u, w_t, b_t, w_o, b_o


def workload(uq, mask, w_u, b_u, w_t, b_t, w_o, b_o):
    q_tanh = jnp.tanh(uq @ w_u + b_u)                 # (S, B, O)
    q_s = jnp.transpose((q_tanh @ w_t + b_t)[..., 0])  # (B, S)
    # masked_softmax: mask applied before exp and again after, eps in denominator
    m = mask.astype(q_s.dtype)
    x = q_s * m
    e_x = jnp.exp(x - jnp.max(x, axis=1, keepdims=True)) * m
    alpha = e_x / (jnp.sum(e_x, axis=1, keepdims=True) + CONFIG['softmax_eps'])
    rq = jnp.einsum('bs,sbi->bi', alpha, uq)           # (B, I)
    return jnp.tanh(rq @ w_o + b_o)
