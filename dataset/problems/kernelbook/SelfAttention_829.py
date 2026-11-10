"""SelfAttention_829: linear Q/K/V projections -> head reshape -> scaled dot product -> softmax -> bmm with V -> head merge -> linear unify."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SelfAttention_829',
    'batch': 64,
    'seq_len': 512,
    'dim': 512,
    'heads': 8,
    'qk_scale_exponent': 0.25,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    b = CONFIG['batch']
    t = CONFIG['seq_len']
    k = CONFIG['dim']
    h = CONFIG['heads']
    scale = CONFIG['weight_scale']
    key = jax.random.key(0)
    k_x, k_q, k_k, k_v, k_u = jax.random.split(key, 5)
    x = jax.random.uniform(k_x, (b, t, k), dtype)
    w_q = jax.random.normal(k_q, (k, k * h), dtype) * scale
    w_k = jax.random.normal(k_k, (k, k * h), dtype) * scale
    w_v = jax.random.normal(k_v, (k, k * h), dtype) * scale
    w_unify = jax.random.normal(k_u, (k * h, k), dtype) * scale
    return x, w_q, w_k, w_v, w_unify


def workload(x, w_q, w_k, w_v, w_unify):
    b, t, k = x.shape
    h = CONFIG['heads']
    Q = (x @ w_q).reshape(b, t, h, k)
    K = (x @ w_k).reshape(b, t, h, k)
    V = (x @ w_v).reshape(b, t, h, k)
    Q = jnp.swapaxes(Q, 1, 2).reshape(b * h, t, k)
    K = jnp.swapaxes(K, 1, 2).reshape(b * h, t, k)
    V = jnp.swapaxes(V, 1, 2).reshape(b * h, t, k)
    scale = k ** CONFIG['qk_scale_exponent']
    Q = Q / scale
    K = K / scale
    dot = Q @ jnp.swapaxes(K, 1, 2)
    dot = jax.nn.softmax(dot, axis=2)
    out = (dot @ V).reshape(b, h, t, k)
    out = jnp.swapaxes(out, 1, 2).reshape(b, t, h * k)
    return out @ w_unify
