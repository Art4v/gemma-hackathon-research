"""W8A16 per-channel dequantize and matrix multiplication with fp32 accumulation."""

import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'w8a16_per_channel_dequant_matmul',
    'M': 4096,
    'K': 8192,
    'N': 28672,
    'qmax': 127,
}


def create_inputs(dtype=jnp.bfloat16):
    k_act, k_w, k_s = jax.random.split(jax.random.key(0), 3)
    M = CONFIG['M']
    K = CONFIG['K']
    N = CONFIG['N']
    qmax = CONFIG['qmax']

    activations = jax.random.normal(k_act, (M, K), dtype=dtype)
    qweight = jax.random.randint(k_w, (K, N), minval=-qmax, maxval=qmax + 1, dtype=jnp.int8)
    scales = jnp.abs(jax.random.normal(k_s, (N,), dtype=dtype)) * 0.02 + 1e-4
    return activations, qweight, scales


def workload(activations, qweight, scales):
    orig_dtype = activations.dtype
    w_dequant = qweight.astype(orig_dtype) * scales[None, :]
    out = jnp.dot(
        activations,
        w_dequant,
        preferred_element_type=jnp.float32,
    )
    return out.astype(orig_dtype)
