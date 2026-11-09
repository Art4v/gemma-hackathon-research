"""[bf16 variant] Fused QKV linear -> multi-head softmax attention -> out linear -> residual add -> LayerNorm."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'MultiHeadAttention_4530',
    'batch': 8,
    'seq_len': 2048,
    'd_model': 1024,
    'n_heads': 4,
    'd_head': 256,
    'eps': 1e-5,
}


def create_inputs(dtype=jnp.bfloat16):
    b, l, d = CONFIG['batch'], CONFIG['seq_len'], CONFIG['d_model']
    hd = CONFIG['n_heads'] * CONFIG['d_head']
    k0, k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 5)
    x = jax.random.uniform(k0, (b, l, d), dtype)
    w_qkv = jax.random.normal(k1, (d, 3 * hd), dtype) * 0.02
    b_qkv = jax.random.normal(k2, (3 * hd,), dtype) * 0.02
    w_out = jax.random.normal(k3, (hd, d), dtype) * 0.02
    b_out = jax.random.normal(k4, (d,), dtype) * 0.02
    gamma = jnp.ones((d,), dtype)
    beta = jnp.zeros((d,), dtype)
    return x, w_qkv, b_qkv, w_out, b_out, gamma, beta


def workload(x, w_qkv, b_qkv, w_out, b_out, gamma, beta):
    bs, seq, _ = x.shape
    n_heads = CONFIG['n_heads']
    # d_head from the weight shape so the port also runs at the original tiny sizes.
    d_head = w_qkv.shape[1] // (3 * n_heads)
    qkv = x @ w_qkv + b_qkv
    q, k, v = jnp.split(qkv, 3, axis=-1)
    q, k, v = (t.reshape(bs, seq, n_heads, d_head).transpose(0, 2, 1, 3) for t in (q, k, v))
    scores = jnp.einsum('bhqd,bhkd->bhqk', q, k) / (d_head ** 0.5)
    probs = jax.nn.softmax(scores, axis=-1)
    attn = jnp.einsum('bhqk,bhkd->bhqd', probs, v)
    attn = attn.transpose(0, 2, 1, 3).reshape(bs, seq, n_heads * d_head)
    y = x + attn @ w_out + b_out
    mean = jnp.mean(y, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(y - mean), axis=-1, keepdims=True)
    return (y - mean) / jnp.sqrt(var + CONFIG['eps']) * gamma + beta
