"""[bf16 variant] GPT-2 self-attention: linear QKV projection, head split, scaled dot-product softmax attention, head merge, linear output projection."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SelfAttentionGPT2_9334',
    'batch': 16,
    'seq_len': 1024,
    'emb': 1024,
    'heads': 8,
    'qkv_factor': 3,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_wa, k_ba, k_wp, k_bp = jax.random.split(jax.random.key(0), 5)
    emb = CONFIG['emb']
    qkv_dim = CONFIG['qkv_factor'] * emb
    scale = CONFIG['weight_scale']
    x = jax.random.uniform(k_x, (CONFIG['batch'], CONFIG['seq_len'], emb), dtype)
    w_attn = jax.random.normal(k_wa, (emb, qkv_dim), dtype) * scale
    b_attn = jax.random.normal(k_ba, (qkv_dim,), dtype) * scale
    w_proj = jax.random.normal(k_wp, (emb, emb), dtype) * scale
    b_proj = jax.random.normal(k_bp, (emb,), dtype) * scale
    return x, w_attn, b_attn, w_proj, b_proj


def workload(x, w_attn, b_attn, w_proj, b_proj):
    b, t, e = x.shape
    # Clamp lets the original emb=4, heads=4 model run against the scaled CONFIG.
    heads = min(CONFIG['heads'], e)
    head_dim = e // heads

    qkv = x @ w_attn + b_attn
    q, k, v = jnp.split(qkv, CONFIG['qkv_factor'], axis=-1)

    q = q.reshape(b, t, heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(b, t, heads, head_dim).transpose(0, 2, 3, 1)
    v = v.reshape(b, t, heads, head_dim).transpose(0, 2, 1, 3)

    dot = (q @ k) / float(head_dim) ** 0.5
    attn = jax.nn.softmax(dot, axis=-1)
    a = attn @ v

    a = a.transpose(0, 2, 1, 3).reshape(b, t, e)
    return a @ w_proj + b_proj
