"""Multi-head attention: QKV proj -> scaled dot-product softmax attention -> fc proj -> residual add -> LayerNorm."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MultiHeadAttention_14276',
    'batch': 8,
    'seq_len': 1024,
    'd_model': 1024,
    'n_head': 16,
    'd_k': 64,
    'd_v': 64,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.float32):
    c = CONFIG
    b, l, d, h, dk, dv = c['batch'], c['seq_len'], c['d_model'], c['n_head'], c['d_k'], c['d_v']
    keys = jax.random.split(jax.random.key(0), 7)
    q = jax.random.uniform(keys[0], (b, l, d), dtype)
    k = jax.random.uniform(keys[1], (b, l, d), dtype)
    v = jax.random.uniform(keys[2], (b, l, d), dtype)
    w_q = jax.random.normal(keys[3], (d, h, dk), dtype) * 0.02
    w_k = jax.random.normal(keys[4], (d, h, dk), dtype) * 0.02
    w_v = jax.random.normal(keys[5], (d, h, dv), dtype) * 0.02
    w_fc = jax.random.normal(keys[6], (h, dv, d), dtype) * 0.02
    gamma = jnp.ones((d,), dtype)
    beta = jnp.zeros((d,), dtype)
    return q, k, v, w_q, w_k, w_v, w_fc, gamma, beta


def workload(q, k, v, w_q, w_k, w_v, w_fc, gamma, beta):
    # Head count and head dims come from the per-head weight layout.
    d_k = w_q.shape[-1]
    qh = jnp.einsum('bld,dhk->bhlk', q, w_q)
    kh = jnp.einsum('bld,dhk->bhlk', k, w_k)
    vh = jnp.einsum('bld,dhk->bhlk', v, w_v)
    scores = jnp.einsum('bhqk,bhsk->bhqs', qh / (d_k ** 0.5), kh)
    attn = jax.nn.softmax(scores, axis=-1)
    ctx = jnp.einsum('bhqs,bhsk->bhqk', attn, vh)
    x = jnp.einsum('bhqk,hkd->bqd', ctx, w_fc) + q
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    out = (x - mean) * jax.lax.rsqrt(var + CONFIG['eps']) * gamma + beta
    return out, attn
