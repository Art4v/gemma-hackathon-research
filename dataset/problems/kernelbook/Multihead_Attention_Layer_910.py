"""[bf16 variant] Multihead self-attention over points: 1x1 Conv1d qkv projection -> per-head scaled softmax attention -> 1x1 Conv1d output projection."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'Multihead_Attention_Layer_910',
    'batch': 16,
    'embedding': 512,
    'num_points': 2048,
    'heads': 4,
}


def create_inputs(dtype=jnp.bfloat16):
    k_x, k_qkv, k_out = jax.random.split(jax.random.key(0), 3)
    b, c, n = CONFIG['batch'], CONFIG['embedding'], CONFIG['num_points']
    x = jax.random.uniform(k_x, (b, c, n), dtype)
    w_qkv = jax.random.normal(k_qkv, (c, 3 * c), dtype) * 0.02
    w_out = jax.random.normal(k_out, (c, c), dtype) * 0.02
    return x, w_qkv, w_out


def workload(x, w_qkv, w_out):
    b, c, n = x.shape
    heads = CONFIG['heads']
    hd = c // heads
    xt = jnp.transpose(x, (0, 2, 1))                      # (B, N, C)
    qkv = xt @ w_qkv                                      # (B, N, 3C)
    qkv = qkv.reshape(b, n, heads, 3 * hd).transpose(0, 2, 1, 3)  # (B, H, N, 3hd)
    q = qkv[..., :hd]
    k = qkv[..., hd:2 * hd]
    v = qkv[..., 2 * hd:]
    scores = jnp.einsum('bhnd,bhmd->bhnm', q, k) / jnp.sqrt(jnp.asarray(c / heads, x.dtype))
    weights = jax.nn.softmax(scores, axis=-1)
    attn = jnp.einsum('bhnm,bhmd->bhnd', weights, v)      # (B, H, N, hd)
    attn = attn.transpose(0, 2, 1, 3).reshape(b, n, c)    # (B, N, C)
    out = attn @ w_out                                    # (B, N, C)
    return jnp.transpose(out, (0, 2, 1))                  # (B, C, N)
