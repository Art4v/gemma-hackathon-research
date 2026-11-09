"""SimpleAttention: linear score -> softmax over seq -> attention-weighted sum of M."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SimpleAttention_60',
    'seq_len': 1024,
    'batch': 32,
    'input_dim': 1024,
}


def create_inputs(dtype=jnp.float32):
    k_m, k_w = jax.random.split(jax.random.key(0))
    M = jax.random.uniform(k_m, (CONFIG['seq_len'], CONFIG['batch'], CONFIG['input_dim']), dtype)
    w = jax.random.normal(k_w, (CONFIG['input_dim'], 1), dtype) * 0.02
    return M, w


def workload(M, w):
    # M: (S, B, D), w: (D, 1)
    scale = M @ w                                  # (S, B, 1)
    alpha = jax.nn.softmax(scale, axis=0)          # softmax over seq
    alpha = jnp.transpose(alpha, (1, 2, 0))        # (B, 1, S)
    attn_pool = jnp.einsum('bos,sbd->bod', alpha, M)[:, 0, :]  # (B, D)
    return attn_pool, alpha
