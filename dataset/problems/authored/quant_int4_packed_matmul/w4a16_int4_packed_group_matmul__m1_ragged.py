"""[m1 M3_ragged variant of authored/quant_int4_packed_matmul/w4a16_int4_packed_group_matmul] W4A16 int4-packed per-group dequant matmul with a ragged token count M=8150 (not a multiple of 8 or 128)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'w4a16_int4_packed_group_matmul_m1_ragged',
    'M': 8150,          # tokens (ragged: not a multiple of 8 or 128)
    'K': 4096,          # hidden size (reduction dim)
    'N': 14336,         # intermediate size (output dim)
    'group_size': 128,  # quantization group along K
    'pack_factor': 2,   # int4 values per int8 byte
    'zero_max': 16,     # zero points in [0, 16)
    'scale_lo': 0.004,
    'scale_hi': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    gs, pf = CONFIG['group_size'], CONFIG['pack_factor']
    G = K // gs
    key = jax.random.key(0)
    kx, kw, ks, kz = jax.random.split(key, 4)
    x = jax.random.normal(kx, (M, K), dtype=jnp.float32).astype(dtype)
    w_packed = jax.random.randint(kw, (K // pf, N), -128, 128, dtype=jnp.int32).astype(jnp.int8)
    scales = jax.random.uniform(ks, (G, N), dtype=jnp.float32,
                                minval=CONFIG['scale_lo'], maxval=CONFIG['scale_hi']).astype(dtype)
    zeros = jax.random.randint(kz, (G, N), 0, CONFIG['zero_max'], dtype=jnp.int32).astype(jnp.int8)
    return x, w_packed, scales, zeros


def workload(x, w_packed, scales, zeros):
    gs = CONFIG['group_size']
    Kp, N = w_packed.shape
    K = Kp * CONFIG['pack_factor']
    G = K // gs

    u = lax.bitcast_convert_type(w_packed, jnp.uint8)
    lo = jnp.bitwise_and(u, jnp.uint8(0x0F))
    hi = jnp.bitwise_and(jnp.right_shift(u, jnp.uint8(4)), jnp.uint8(0x0F))
    # Row 2i = low nibble, row 2i+1 = high nibble.
    q = jnp.stack([lo, hi], axis=1).reshape(K, N).astype(jnp.float32)

    q = q.reshape(G, gs, N)
    z = zeros.astype(jnp.float32)[:, None, :]
    s = scales.astype(jnp.float32)[:, None, :]
    w = ((q - z) * s).reshape(K, N).astype(x.dtype)

    out = jnp.dot(x, w, preferred_element_type=jnp.float32)
    return out.astype(x.dtype)
