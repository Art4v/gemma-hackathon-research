"""Weight-only int8 per-channel dequantization fused with bfloat16 matrix multiplication."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'w8a16_per_channel_dequant_matmul',
    'M': 4096,
    'K': 8192,
    'N': 28672,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    M = CONFIG['M']
    K = CONFIG['K']
    N = CONFIG['N']
    x = jax.random.normal(k1, (M, K), dtype=dtype)
    w_q = jax.random.randint(k2, (K, N), minval=-128, maxval=127, dtype=jnp.int8)
    scales = (jax.random.uniform(k3, (N,), dtype=jnp.float32, minval=0.5, maxval=1.5) * (0.02 / 127.0)).astype(dtype)
    return x, w_q, scales


def workload(x, w_q, scales):
    w_dequant = w_q.astype(jnp.float32) * scales.astype(jnp.float32)[None, :]
    acc = jnp.dot(x.astype(jnp.float32), w_dequant, preferred_element_type=jnp.float32)
    return acc.astype(x.dtype)
