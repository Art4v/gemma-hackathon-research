"""SoftAttention: tanh(Linear(x)) @ v -> softmax -> attention-weighted sum over dim 1."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SoftAttention_16399',
    'batch': 8,
    'd1': 16,
    'd2': 512,  # must equal emb_dim: torch's expand_as broadcasts att's last axis against emb
    'emb_dim': 512,
    'attn_dim': 512,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw, kb, kv = jax.random.split(key, 4)
    c = CONFIG
    values = jax.random.uniform(kx, (c['batch'], c['d1'], c['d2'], c['emb_dim']), dtype)
    w = jax.random.normal(kw, (c['emb_dim'], c['attn_dim']), dtype) * 0.02
    b = jax.random.normal(kb, (c['attn_dim'],), dtype) * 0.02
    v = jax.random.normal(kv, (c['attn_dim'],), dtype) * 0.02
    return values, w, b, v


def workload(values, w, b, v):
    h = jnp.tanh(values @ w + b)
    e = h @ v
    att = jax.nn.softmax(jnp.squeeze(e, axis=1) if e.shape[1] == 1 else e, axis=-1)
    # torch: values.transpose(1,2) * att.unsqueeze(1).expand_as(...), summed over dim 2
    # == out[b,i,k] = sum_j values[b,j,i,k] * att[b,j,k]
    out = jnp.sum(values * att[:, :, None, :], axis=1)
    return jnp.squeeze(out)
