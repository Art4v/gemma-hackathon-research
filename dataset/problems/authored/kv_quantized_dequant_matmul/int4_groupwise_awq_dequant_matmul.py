"""Unpack int4 (packed in uint8) -> group-wise asymmetric dequant (zero point + scale) -> bf16 matmul with f32 accumulation."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'int4_groupwise_awq_dequant_matmul',
    'M': 2048,          # tokens (prefill batch * seq)
    'K': 8192,          # hidden size (Llama-3-70B)
    'N': 28672,         # MLP intermediate size (Llama-3-70B)
    'group_size': 128,  # quantization group along K
    'bits': 4,
    'scale_std': 0.02,
    'scale_min': 0.001,
}


def create_inputs(dtype=jnp.bfloat16):
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    G = CONFIG['group_size']
    qmax = 2 ** CONFIG['bits']
    key = jax.random.key(0)
    kx, kw, ks, kz = jax.random.split(key, 4)
    x = jax.random.normal(kx, (M, K), dtype=jnp.float32).astype(dtype)
    # Two 4-bit values per byte: low nibble = row 2i, high nibble = row 2i+1.
    packed_w = jax.random.randint(kw, (K // 2, N), 0, qmax * qmax, dtype=jnp.int32).astype(jnp.uint8)
    scales = (jnp.abs(jax.random.normal(ks, (K // G, N), dtype=jnp.float32)) * CONFIG['scale_std']
              + CONFIG['scale_min']).astype(dtype)
    zeros = jax.random.randint(kz, (K // G, N), 0, qmax, dtype=jnp.int32).astype(jnp.uint8)
    return x, packed_w, scales, zeros


def workload(x, packed_w, scales, zeros):
    K = x.shape[1]
    N = packed_w.shape[1]
    G = CONFIG['group_size']
    bits = CONFIG['bits']
    mask = (1 << bits) - 1

    p = packed_w.astype(jnp.int32)
    lo = jnp.bitwise_and(p, mask)
    hi = jnp.bitwise_and(jnp.right_shift(p, bits), mask)
    q = jnp.stack([lo, hi], axis=1).reshape(K, N)  # (K, N) int32 in [0, 2^bits)

    qg = q.reshape(K // G, G, N).astype(jnp.float32)
    z = zeros.astype(jnp.float32)[:, None, :]
    s = scales.astype(jnp.float32)[:, None, :]
    w = ((qg - z) * s).reshape(K, N).astype(x.dtype)

    y = lax.dot_general(
        x, w,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    return y.astype(x.dtype)
