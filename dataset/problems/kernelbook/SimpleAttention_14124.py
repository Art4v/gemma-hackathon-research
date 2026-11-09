"""SimpleAttention: linear score -> softmax over seq -> attention-weighted pooling."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SimpleAttention_14124',
    'seq_len': 1024,
    'batch': 32,
    'input_dim': 1024,
}


def create_inputs(dtype=jnp.float32):
    k_m, k_w = jax.random.split(jax.random.key(0))
    m = jax.random.uniform(
        k_m, (CONFIG['seq_len'], CONFIG['batch'], CONFIG['input_dim']), dtype)
    w = jax.random.normal(k_w, (CONFIG['input_dim'], 1), dtype) * 0.02
    return m, w


def workload(m, w):
    scale = jnp.einsum('sbd,do->sbo', m, w)          # (S, B, 1)
    alpha = jax.nn.softmax(scale, axis=0)
    alpha = jnp.transpose(alpha, (1, 2, 0))          # (B, 1, S)
    attn_pool = jnp.einsum('bs,sbd->bd', alpha[:, 0, :], m)
    return attn_pool, alpha
