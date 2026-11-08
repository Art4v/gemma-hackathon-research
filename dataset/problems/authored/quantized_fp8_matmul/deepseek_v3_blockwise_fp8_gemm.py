"""DeepSeek-V3 style 128x128 blockwise-scaled FP8 (float8_e4m3fn) matrix multiplication."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'deepseek_v3_blockwise_fp8_gemm',
    'M': 8192,
    'K': 7168,
    'N': 14336,
    'block_m': 128,
    'block_k': 128,
    'block_n': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    M = CONFIG['M']
    K = CONFIG['K']
    N = CONFIG['N']
    bm = CONFIG['block_m']
    bk = CONFIG['block_k']
    bn = CONFIG['block_n']

    k0, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    a_fp8 = jax.random.normal(k0, (M, K), dtype=jnp.float32).astype(jnp.float8_e4m3fn)
    b_fp8 = (jax.random.normal(k1, (K, N), dtype=jnp.float32) * 0.02).astype(jnp.float8_e4m3fn)

    a_scale = jax.random.uniform(k2, (M // bm, K // bk), dtype=jnp.float32, minval=0.001, maxval=0.05)
    b_scale = jax.random.uniform(k3, (K // bk, N // bn), dtype=jnp.float32, minval=0.001, maxval=0.05)
    return a_fp8, b_fp8, a_scale, b_scale


def workload(a_fp8, b_fp8, a_scale, b_scale):
    M, K = a_fp8.shape
    _, N = b_fp8.shape
    bm = CONFIG['block_m']
    bk = CONFIG['block_k']
    bn = CONFIG['block_n']

    a_blocks = a_fp8.astype(jnp.float32).reshape(M // bm, bm, K // bk, bk)
    a_deq = (a_blocks * a_scale[:, None, :, None]).reshape(M, K)

    b_blocks = b_fp8.astype(jnp.float32).reshape(K // bk, bk, N // bn, bn)
    b_deq = (b_blocks * b_scale[:, None, :, None]).reshape(K, N)

    out = jnp.dot(a_deq, b_deq, preferred_element_type=jnp.float32)
    return out.astype(jnp.bfloat16)
