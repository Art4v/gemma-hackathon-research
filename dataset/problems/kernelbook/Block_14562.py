"""[bf16 variant] ViT block: x + proj(softmax(LN1(x) Wq (LN1(x) Wk)^T * scale) LN1(x) Wv), then x + fc2(gelu(fc1(LN2(x))))."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Block_14562',
    'batch': 16,
    'seq_len': 1024,
    'dim': 768,
    'num_heads': 12,
    'mlp_hidden_dim': 3072,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    b, n, c = CONFIG['batch'], CONFIG['seq_len'], CONFIG['dim']
    h, m = CONFIG['num_heads'], CONFIG['mlp_hidden_dim']
    ks = jax.random.split(jax.random.key(0), 8)
    x = jax.random.normal(ks[0], (b, n, c), dtype)
    ln1_g = jnp.ones((c,), dtype)
    ln1_b = jnp.zeros((c,), dtype)
    # qkv weight as (in, 3, heads, head_dim): same split order as torch's reshape(B, N, 3, H, D)
    w_qkv = jax.random.normal(ks[1], (c, 3, h, c // h), dtype) * 0.02
    w_proj = jax.random.normal(ks[2], (c, c), dtype) * 0.02
    b_proj = jax.random.normal(ks[3], (c,), dtype) * 0.02
    ln2_g = jnp.ones((c,), dtype)
    ln2_b = jnp.zeros((c,), dtype)
    w_fc1 = jax.random.normal(ks[4], (c, m), dtype) * 0.02
    b_fc1 = jax.random.normal(ks[5], (m,), dtype) * 0.02
    w_fc2 = jax.random.normal(ks[6], (m, c), dtype) * 0.02
    b_fc2 = jax.random.normal(ks[7], (c,), dtype) * 0.02
    return (x, ln1_g, ln1_b, w_qkv, w_proj, b_proj,
            ln2_g, ln2_b, w_fc1, b_fc1, w_fc2, b_fc2)


def _layer_norm(x, g, b):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + CONFIG['eps']) * g + b


def workload(x, ln1_g, ln1_b, w_qkv, w_proj, b_proj,
             ln2_g, ln2_b, w_fc1, b_fc1, w_fc2, b_fc2):
    bsz, n, c = x.shape
    head_dim = w_qkv.shape[3]
    h = _layer_norm(x, ln1_g, ln1_b)
    qkv = jnp.einsum('bnc,cthd->tbhnd', h, w_qkv)
    q, k, v = qkv[0], qkv[1], qkv[2]
    attn = jnp.einsum('bhqd,bhkd->bhqk', q, k) * (head_dim ** -0.5)
    attn = jax.nn.softmax(attn, axis=-1)
    o = jnp.einsum('bhqk,bhkd->bhqd', attn, v)
    o = jnp.transpose(o, (0, 2, 1, 3)).reshape(bsz, n, c)
    x = x + (o @ w_proj + b_proj)
    h2 = _layer_norm(x, ln2_g, ln2_b)
    mlp = jax.nn.gelu(h2 @ w_fc1 + b_fc1, approximate=False) @ w_fc2 + b_fc2
    return x + mlp
