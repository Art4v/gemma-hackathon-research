"""Int8 weight-only fused QKV projection: int8->bf16 dequant matmul (f32 acc) * per-channel scale + bias -> split into GQA Q/K/V heads."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'int8_fused_qkv_bias_proj',
    'num_tokens': 4096,
    'hidden': 5120,
    'num_q_heads': 40,
    'num_kv_heads': 8,
    'head_dim': 128,
    'scale_min': 1e-4,
    'scale_max': 3e-4,
}


def create_inputs(dtype=jnp.float32):
    T = CONFIG['num_tokens']
    H = CONFIG['hidden']
    D = CONFIG['head_dim']
    N = (CONFIG['num_q_heads'] + 2 * CONFIG['num_kv_heads']) * D
    key = jax.random.key(0)
    kx, kw, ks, kb = jax.random.split(key, 4)
    x = jax.random.normal(kx, (T, H), dtype)
    w_q = jax.random.randint(kw, (H, N), -127, 128, dtype=jnp.int32).astype(jnp.int8)
    scales = jax.random.uniform(ks, (N,), jnp.float32,
                                minval=CONFIG['scale_min'], maxval=CONFIG['scale_max'])
    bias = jax.random.normal(kb, (N,), dtype) * 0.02
    return x, w_q, scales, bias


def workload(x, w_q, scales, bias):
    D = CONFIG['head_dim']
    nq = CONFIG['num_q_heads']
    nkv = CONFIG['num_kv_heads']
    T = x.shape[0]
    # int8 values are exactly representable in bf16; scale applied per output channel after f32 accumulation.
    w = w_q.astype(x.dtype)
    acc = jnp.dot(x, w, preferred_element_type=jnp.float32)
    out = acc * scales.astype(jnp.float32)[None, :] + bias.astype(jnp.float32)[None, :]
    out = out.astype(x.dtype)
    q_end = nq * D
    k_end = q_end + nkv * D
    q = out[:, :q_end].reshape(T, nq, D)
    k = out[:, q_end:k_end].reshape(T, nkv, D)
    v = out[:, k_end:].reshape(T, nkv, D)
    return q, k, v
