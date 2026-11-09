"""AttentionUnit: xEmbed(x) + sEmbed(sPrev) -> tanh -> wEmbed -> softmax over T."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'AttentionUnit_14625',
    'batch': 64,
    'seq_len': 1024,
    'x_dim': 1024,
    's_dim': 1024,
    'att_dim': 1024,
}


def create_inputs(dtype=jnp.float32):
    keys = jax.random.split(jax.random.key(0), 8)
    b, t = CONFIG['batch'], CONFIG['seq_len']
    xd, sd, ad = CONFIG['x_dim'], CONFIG['s_dim'], CONFIG['att_dim']
    x = jax.random.uniform(keys[0], (b, t, xd), dtype)
    s_prev = jax.random.uniform(keys[1], (b, sd), dtype)
    w_s = jax.random.normal(keys[2], (sd, ad), dtype) * 0.02
    b_s = jax.random.normal(keys[3], (ad,), dtype) * 0.02
    w_x = jax.random.normal(keys[4], (xd, ad), dtype) * 0.02
    b_x = jax.random.normal(keys[5], (ad,), dtype) * 0.02
    w_v = jax.random.normal(keys[6], (ad, 1), dtype) * 0.02
    b_v = jax.random.normal(keys[7], (1,), dtype) * 0.02
    return x, s_prev, w_s, b_s, w_x, b_x, w_v, b_v


def workload(x, s_prev, w_s, b_s, w_x, b_x, w_v, b_v):
    x_proj = x @ w_x + b_x                      # (B, T, A)
    s_proj = (s_prev @ w_s + b_s)[:, None, :]   # (B, 1, A), broadcast over T
    sum_tanh = jnp.tanh(s_proj + x_proj)
    v_proj = (sum_tanh @ w_v + b_v)[..., 0]     # (B, T)
    return jax.nn.softmax(v_proj, axis=1)
