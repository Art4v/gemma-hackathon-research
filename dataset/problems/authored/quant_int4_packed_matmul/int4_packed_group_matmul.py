"""Int4 packed weight-only quantized matmul: unpack nibbles -> per-group (q - zero) * scale dequant -> bf16 matmul with f32 accumulation."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'int4_packed_group_matmul',
    'M': 2048,
    'K': 8192,
    'N': 14336,
    'group_size': 128,
    'bits': 4,
    'scale_lo': 0.004,
    'scale_hi': 0.012,
}


def create_inputs(dtype=jnp.float32):
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    G = K // CONFIG['group_size']
    qmax = (1 << CONFIG['bits']) - 1
    k_x, k_w, k_s, k_z = jax.random.split(jax.random.key(0), 4)
    x = jax.random.normal(k_x, (M, K), dtype)
    # Two unsigned int4 values per byte along K: low nibble = row 2i, high nibble = row 2i+1.
    w_packed = jax.random.randint(k_w, (K // 2, N), -128, 128, dtype=jnp.int32).astype(jnp.int8)
    scales = jax.random.uniform(
        k_s, (G, N), jnp.float32, CONFIG['scale_lo'], CONFIG['scale_hi']
    ).astype(dtype)
    zeros = jax.random.randint(k_z, (G, N), 0, qmax + 1, dtype=jnp.int32).astype(jnp.int8)
    return x, w_packed, scales, zeros


def workload(x, w_packed, scales, zeros):
    Kh, N = w_packed.shape
    K = Kh * 2
    gs = CONFIG['group_size']
    G = K // gs
    bits = CONFIG['bits']
    mask = (1 << bits) - 1

    # Unpack nibbles as unsigned values in [0, 15].
    u = lax.bitcast_convert_type(w_packed, jnp.uint8).astype(jnp.int32)
    lo = jnp.bitwise_and(u, mask)
    hi = jnp.bitwise_and(jnp.right_shift(u, bits), mask)
    q = jnp.stack([lo, hi], axis=1).reshape(K, N)  # interleave along K

    # Per-group asymmetric dequant in f32.
    qg = q.reshape(G, gs, N).astype(jnp.float32)
    z = zeros.astype(jnp.float32)[:, None, :]
    s = scales.astype(jnp.float32)[:, None, :]
    w = ((qg - z) * s).reshape(K, N).astype(x.dtype)

    out = jnp.dot(x, w, preferred_element_type=jnp.float32)
    return out.astype(x.dtype)
