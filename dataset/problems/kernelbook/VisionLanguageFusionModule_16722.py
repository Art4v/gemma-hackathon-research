"""Packed QKV linear projection -> multi-head scaled dot-product cross-attention -> output linear -> element-wise product with tgt."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'VisionLanguageFusionModule_16722',
    'tgt_len': 1024,
    'memory_len': 2048,
    'batch': 8,
    'd_model': 512,
    'nhead': 4,
}


def create_inputs(dtype=jnp.float32):
    L = CONFIG['tgt_len']
    S = CONFIG['memory_len']
    N = CONFIG['batch']
    E = CONFIG['d_model']
    keys = jax.random.split(jax.random.key(0), 6)
    tgt = jax.random.uniform(keys[0], (L, N, E), dtype)
    memory = jax.random.uniform(keys[1], (S, N, E), dtype)
    in_proj_w = jax.random.normal(keys[2], (E, 3 * E), dtype) * 0.02
    in_proj_b = jax.random.normal(keys[3], (3 * E,), dtype) * 0.02
    out_proj_w = jax.random.normal(keys[4], (E, E), dtype) * 0.02
    out_proj_b = jax.random.normal(keys[5], (E,), dtype) * 0.02
    return tgt, memory, in_proj_w, in_proj_b, out_proj_w, out_proj_b


def workload(tgt, memory, in_proj_w, in_proj_b, out_proj_w, out_proj_b):
    E = tgt.shape[-1]
    h = CONFIG['nhead']
    d = E // h
    scale = (1.0 / d) ** 0.5

    q = tgt @ in_proj_w[:, :E] + in_proj_b[:E]
    kv = memory @ in_proj_w[:, E:] + in_proj_b[E:]
    k, v = jnp.split(kv, 2, axis=-1)

    # (L, [N], E) -> ([N], h, L, d)
    q = jnp.moveaxis(q.reshape(q.shape[:-1] + (h, d)), 0, -2)
    k = jnp.moveaxis(k.reshape(k.shape[:-1] + (h, d)), 0, -2)
    v = jnp.moveaxis(v.reshape(v.shape[:-1] + (h, d)), 0, -2)

    scores = (q * scale) @ jnp.swapaxes(k, -1, -2)
    weights = jax.nn.softmax(scores, axis=-1)
    attn = weights @ v

    # ([N], h, L, d) -> (L, [N], E)
    attn = jnp.moveaxis(attn, -2, 0)
    attn = attn.reshape(attn.shape[:-2] + (E,))

    out = attn @ out_proj_w + out_proj_b
    return tgt * out
