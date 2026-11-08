"""DeepSeek-V3 fine-grained 128x128 blockwise-scaled FP8 (float8_e4m3fn) matrix multiplication."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'deepseek_v3_blockwise_fp8_gemm',
    'M': 4096,
    'K': 7168,
    'N': 7168,
    'block_m': 128,
    'block_k': 128,
    'block_n': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    m = CONFIG['M']
    k = CONFIG['K']
    n = CONFIG['N']
    bm = CONFIG['block_m']
    bk = CONFIG['block_k']
    bn = CONFIG['block_n']

    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    a_raw = jax.random.normal(k1, (m, k), dtype=jnp.float32)
    b_raw = jax.random.normal(k2, (k, n), dtype=jnp.float32)
    a_fp8 = a_raw.astype(jnp.float8_e4m3fn)
    b_fp8 = b_raw.astype(jnp.float8_e4m3fn)

    a_scale = jax.random.uniform(k3, (m // bm, k // bk), dtype=jnp.float32, minval=0.005, maxval=0.05)
    b_scale = jax.random.uniform(k4, (k // bk, n // bn), dtype=jnp.float32, minval=0.001, maxval=0.02)

    return a_fp8, b_fp8, a_scale, b_scale


def workload(a_fp8, b_fp8, a_scale, b_scale):
    m, k = a_fp8.shape
    _, n = b_fp8.shape
    bm = CONFIG['block_m']
    bk = CONFIG['block_k']
    bn = CONFIG['block_n']

    a_f32 = a_fp8.astype(jnp.float32).reshape(m // bm, bm, k // bk, bk)
    a_deq = (a_f32 * a_scale[:, None, :, None]).reshape(m, k)

    b_f32 = b_fp8.astype(jnp.float32).reshape(k // bk, bk, n // bn, bn)
    b_deq = (b_f32 * b_scale[:, None, :, None]).reshape(k, n)

    out = jnp.dot(a_deq, b_deq, preferred_element_type=jnp.float32)
    return out.astype(jnp.bfloat16)
