"""Fused multi-head attention with alternating sliding-window local and full causal global heads and logit soft-capping."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'hybrid_local_global_head_attention',
    'batch_size': 4,
    'seq_len': 4096,
    'num_heads': 32,
    'head_dim': 128,
    'window_size': 512,
    'softcap': 50.0,
}

def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    B = CONFIG['batch_size']
    S = CONFIG['seq_len']
    H = CONFIG['num_heads']
    D = CONFIG['head_dim']
    q = jax.random.normal(k1, (B, H, S, D), dtype=dtype)
    k = jax.random.normal(k2, (B, H, S, D), dtype=dtype)
    v = jax.random.normal(k3, (B, H, S, D), dtype=dtype)
    return (q, k, v)

def workload(q, k, v):
    B, H, S, D = q.shape
    W = CONFIG['window_size']
    softcap = CONFIG['softcap']
    scale = 1.0 / jnp.sqrt(jnp.float32(D))

    logits = jnp.einsum('bhsd,bhtd->bhst', q.astype(jnp.float32), k.astype(jnp.float32)) * scale
    logits = softcap * jnp.tanh(logits / softcap)

    q_idx = jnp.arange(S)[:, None]
    k_idx = jnp.arange(S)[None, :]

    causal_mask = q_idx >= k_idx
    local_mask = causal_mask & ((q_idx - k_idx) < W)

    head_is_global = (jnp.arange(H) % 2) == 1
    mask = jnp.where(head_is_global[:, None, None], causal_mask[None, :, :], local_mask[None, :, :])

    neg_inf = jnp.finfo(jnp.float32).min
    logits = jnp.where(mask[None, :, :, :], logits, neg_inf)

    probs = jax.nn.softmax(logits, axis=-1)
    out = jnp.einsum('bhst,bhtd->bhsd', probs, v.astype(jnp.float32))
    return out.astype(q.dtype)
