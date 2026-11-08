"""Int4 (two-per-int8 packed) unpack -> per-group (128) zero-point/scale dequant -> bf16 x W matmul with f32 accumulation (W4A16)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'int4_packed_groupwise_w4a16_matmul',
    'M': 2048,
    'K': 8192,
    'N': 28672,
    'group_size': 128,
    'bits': 4,
    'pack_factor': 2,
    'scale_base': 0.0025,
}


def create_inputs(dtype=jnp.float32):
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    G = K // CONFIG['group_size']
    P = CONFIG['pack_factor']
    qmax = 1 << CONFIG['bits']
    k0, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    x = jax.random.normal(k0, (M, K), dtype=jnp.float32).astype(dtype)
    # Full byte range: each nibble is an unsigned int4 code in [0, 15].
    w_packed = jax.random.randint(k1, (K // P, N), -128, 128, dtype=jnp.int32).astype(jnp.int8)
    scales = (jax.random.uniform(k2, (G, N), dtype=jnp.float32, minval=0.5, maxval=1.5)
              * CONFIG['scale_base']).astype(dtype)
    zeros = jax.random.randint(k3, (G, N), 0, qmax, dtype=jnp.int32).astype(jnp.int8)
    return x, w_packed, scales, zeros


def workload(x, w_packed, scales, zeros):
    bits = CONFIG['bits']
    gs = CONFIG['group_size']
    mask = (1 << bits) - 1
    Kp, N = w_packed.shape
    K = Kp * CONFIG['pack_factor']
    G = K // gs

    # Unpack: low nibble -> even k row, high nibble -> odd k row.
    p = w_packed.astype(jnp.int32)
    lo = p & mask
    hi = (p >> bits) & mask
    q = jnp.stack([lo, hi], axis=1).reshape(K, N)

    # Per-group asymmetric dequant in f32.
    qg = q.reshape(G, gs, N).astype(jnp.float32)
    z = zeros.astype(jnp.float32)[:, None, :]
    s = scales.astype(jnp.float32)[:, None, :]
    w = ((qg - z) * s).reshape(K, N).astype(x.dtype)

    out = lax.dot_general(
        x, w,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    return out.astype(x.dtype)
