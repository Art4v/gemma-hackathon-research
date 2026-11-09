"""Multi-head self-attention: QKV projections, k**-0.25 scaling of q and k, softmax(QK^T)V, unify-heads Linear."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SelfAttention_1864',
    'batch': 8,
    'seq_len': 2048,
    'k': 512,
    'heads': 8,
}


def create_inputs(dtype=jnp.float32):
    b, t, k, h = CONFIG['batch'], CONFIG['seq_len'], CONFIG['k'], CONFIG['heads']
    keys = jax.random.split(jax.random.key(0), 6)
    x = jax.random.uniform(keys[0], (b, t, k), dtype)
    wq = jax.random.normal(keys[1], (k, h * k), dtype) * 0.02
    wk = jax.random.normal(keys[2], (k, h * k), dtype) * 0.02
    wv = jax.random.normal(keys[3], (k, h * k), dtype) * 0.02
    wu = jax.random.normal(keys[4], (h * k, k), dtype) * 0.02
    bu = jax.random.normal(keys[5], (k,), dtype) * 0.02
    return x, wq, wk, wv, wu, bu


def workload(x, wq, wk, wv, wu, bu):
    b, t, k = x.shape
    h = CONFIG['heads']
    scale = k ** 0.25
    q = (x @ wq).reshape(b, t, h, k) / scale
    kk = (x @ wk).reshape(b, t, h, k) / scale
    v = (x @ wv).reshape(b, t, h, k)
    dot = jnp.einsum('bqhd,bkhd->bhqk', q, kk)
    attn = jax.nn.softmax(dot, axis=-1)
    out = jnp.einsum('bhqk,bkhd->bqhd', attn, v).reshape(b, t, h * k)
    return out @ wu + bu
