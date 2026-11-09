"""HighwayMaxoutNetwork: tanh(Linear) -> concat -> Linear+maxout -> Linear+maxout -> Linear -> max -> masked log_softmax."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'HighwayMaxoutNetwork_4005',
    'batch': 32,
    'seq_len': 1024,
    'mod_out_size': 512,
    'hidden_size': 512,
    'max_out_pool_size': 4,
    'mask_fill': -1e30,
}


def create_inputs(dtype=jnp.float32):
    B, L = CONFIG['batch'], CONFIG['seq_len']
    M, H, P = CONFIG['mod_out_size'], CONFIG['hidden_size'], CONFIG['max_out_pool_size']
    ks = jax.random.split(jax.random.key(0), 12)
    mod = jax.random.uniform(ks[0], (B, L, M), dtype)
    h_i = jax.random.uniform(ks[1], (B, H), dtype)
    u_s_prev = jax.random.uniform(ks[2], (B, M), dtype)
    u_e_prev = jax.random.uniform(ks[3], (B, M), dtype)
    mask = jax.random.randint(ks[4], (B, L), 0, 2).astype(dtype)
    w_r = jax.random.normal(ks[5], (2 * M + H, H), dtype) * 0.02
    w1 = jax.random.normal(ks[6], (M + H, P * H), dtype) * 0.02
    b1 = jax.random.normal(ks[7], (P * H,), dtype) * 0.02
    w2 = jax.random.normal(ks[8], (H, P * H), dtype) * 0.02
    b2 = jax.random.normal(ks[9], (P * H,), dtype) * 0.02
    w3 = jax.random.normal(ks[10], (2 * H, P), dtype) * 0.02
    b3 = jax.random.normal(ks[11], (P,), dtype) * 0.02
    return mod, h_i, u_s_prev, u_e_prev, mask, w_r, w1, b1, w2, b2, w3, b3


def workload(mod, h_i, u_s_prev, u_e_prev, mask, w_r, w1, b1, w2, b2, w3, b3):
    B, L, _ = mod.shape
    H = w_r.shape[1]
    P = CONFIG['max_out_pool_size']
    r = jnp.tanh(jnp.concatenate((h_i, u_s_prev, u_e_prev), axis=1) @ w_r)
    r_exp = jnp.broadcast_to(r[:, None, :], (B, L, H))
    w1_inp = jnp.concatenate((mod, r_exp), axis=2)
    # Out features are laid out as (pool, hidden), matching torch .view(B, L, P, H).
    m1 = (w1_inp @ w1 + b1).reshape(B, L, P, H).max(axis=2)
    m2 = (m1 @ w2 + b2).reshape(B, L, P, H).max(axis=2)
    alpha = jnp.concatenate((m1, m2), axis=2) @ w3 + b3
    logits = alpha.max(axis=2)
    mask = mask.astype(jnp.float32)
    masked = mask * logits + (1.0 - mask) * CONFIG['mask_fill']
    return jax.nn.log_softmax(masked, axis=-1)
