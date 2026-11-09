"""[bf16 variant] Self-attention: packed QKV projection -> multi-head scaled dot-product softmax attention -> output projection (nn.MultiheadAttention, seq-first, batch 1)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'SelfAttention_241',
    'seq_len': 2048,
    'embed_dim': 4096,
    'num_heads': 32,
}


def create_inputs(dtype=jnp.bfloat16):
    L, E = CONFIG['seq_len'], CONFIG['embed_dim']
    kx, kwi, kbi, kwo, kbo = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(kx, (L, E), dtype)
    w_in = jax.random.normal(kwi, (E, 3 * E), dtype) * 0.02
    b_in = jax.random.normal(kbi, (3 * E,), dtype) * 0.02
    w_out = jax.random.normal(kwo, (E, E), dtype) * 0.02
    b_out = jax.random.normal(kbo, (E,), dtype) * 0.02
    return x, w_in, b_in, w_out, b_out


def workload(x, w_in, b_in, w_out, b_out):
    L, E = x.shape
    # Clamp so the same code runs on the original torch model (E=4, 4 heads, head_dim 1).
    h = min(CONFIG['num_heads'], E)
    d = E // h
    qkv = x @ w_in + b_in
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q = (q * (d ** -0.5)).reshape(L, h, d).transpose(1, 0, 2)
    k = k.reshape(L, h, d).transpose(1, 0, 2)
    v = v.reshape(L, h, d).transpose(1, 0, 2)
    s = jnp.einsum('hld,hmd->hlm', q, k)
    p = jax.nn.softmax(s, axis=-1)
    o = jnp.einsum('hlm,hmd->hld', p, v).transpose(1, 0, 2).reshape(L, E)
    return o @ w_out + b_out
