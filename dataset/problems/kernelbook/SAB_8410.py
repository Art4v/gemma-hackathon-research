"""SAB set-attention block: linear QKV projections, multi-head scaled dot-product attention, output projection, residual, ReLU feed-forward residual."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SAB_8410',
    'batch': 8,
    'seq_len': 2048,
    'd': 1024,
    'd_p': 1024,
    'n_heads': 4,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    B, N = CONFIG['batch'], CONFIG['seq_len']
    d, d_p = CONFIG['d'], CONFIG['d_p']
    scale = CONFIG['weight_scale']
    keys = jax.random.split(jax.random.key(0), 11)
    x = jax.random.uniform(keys[0], (B, N, d), dtype)
    q_w = jax.random.normal(keys[1], (d, d_p), dtype) * scale
    q_b = jax.random.normal(keys[2], (d_p,), dtype) * scale
    k_w = jax.random.normal(keys[3], (d, d_p), dtype) * scale
    k_b = jax.random.normal(keys[4], (d_p,), dtype) * scale
    v_w = jax.random.normal(keys[5], (d, d_p), dtype) * scale
    v_b = jax.random.normal(keys[6], (d_p,), dtype) * scale
    o_w = jax.random.normal(keys[7], (d_p, d), dtype) * scale
    o_b = jax.random.normal(keys[8], (d,), dtype) * scale
    fc_w = jax.random.normal(keys[9], (d, d), dtype) * scale
    fc_b = jax.random.normal(keys[10], (d,), dtype) * scale
    return x, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b, fc_w, fc_b


def workload(x, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b, fc_w, fc_b):
    B, N, _ = x.shape
    H = CONFIG['n_heads']
    d_h = q_w.shape[1] // H

    def split_heads(p):
        p = p.reshape(B, N, H, d_h)
        p = jnp.transpose(p, (2, 0, 1, 3))
        return p.reshape(H * B, N, d_h)

    q = split_heads(x @ q_w + q_b)
    k = split_heads(x @ k_w + k_b)
    v = split_heads(x @ v_w + v_b)

    routing = q @ jnp.swapaxes(k, 1, 2)
    routing = jax.nn.softmax(routing / jnp.sqrt(jnp.asarray(d_h, x.dtype)), axis=-1)
    o = routing @ v

    o = o.reshape(H, B, N, d_h)
    o = jnp.transpose(o, (1, 2, 0, 3)).reshape(B, N, H * d_h)

    h = o @ o_w + o_b
    h = h + x
    return h + jax.nn.relu(h @ fc_w + fc_b)
