"""W8A16 per-channel weight dequantization and matrix multiplication with float32 accumulation."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'w8a16_per_channel_dequant_matmul',
    'M': 2048,
    'K': 8192,
    'N': 28672,
}


def create_inputs(dtype=jnp.bfloat16):
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    M = CONFIG['M']
    K = CONFIG['K']
    N = CONFIG['N']
    x = jax.random.normal(k1, (M, K), dtype=dtype)
    w_q = jax.random.randint(k2, (K, N), minval=-128, maxval=127, dtype=jnp.int8)
    scales = (jax.random.uniform(k3, (N,), dtype=jnp.float32, minval=0.5, maxval=1.5) * (0.02 / 127.0)).astype(dtype)
    return x, w_q, scales


def workload(x, w_q, scales):
    w_deq = w_q.astype(x.dtype) * scales[None, :]
    out = jnp.dot(x, w_deq, preferred_element_type=jnp.float32)
    return out.astype(x.dtype)
