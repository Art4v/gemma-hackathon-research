"""Weight-only INT4 packed group-quantized matrix multiplication (W4A16) with per-group scales and zero points."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'quant_int4_packed_matmul',
    'M': 2048,
    'K': 8192,
    'N': 28672,
    'group_size': 128,
    'pack_factor': 2,
    'bit_shift': 4,
    'bit_mask': 15,
}

def create_inputs(dtype=jnp.bfloat16):
    k0, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    M = CONFIG['M']
    K = CONFIG['K']
    N = CONFIG['N']
    G = K // CONFIG['group_size']
    K_packed = K // CONFIG['pack_factor']

    x = jax.random.normal(k0, (M, K), dtype=dtype)
    qweight = jax.random.randint(k1, (K_packed, N), minval=0, maxval=256, dtype=jnp.int32).astype(jnp.uint8)
    scales = jax.random.normal(k2, (G, N), dtype=dtype) * 0.02
    zeros = jax.random.uniform(k3, (G, N), dtype=dtype, minval=0.0, maxval=15.0)
    return x, qweight, scales, zeros

def workload(x, qweight, scales, zeros):
    M, K = x.shape
    K_packed, N = qweight.shape
    G = scales.shape[0]
    group_size = CONFIG['group_size']
    pack_factor = CONFIG['pack_factor']
    shift = jnp.uint8(CONFIG['bit_shift'])
    mask = jnp.uint8(CONFIG['bit_mask'])

    w_low = jnp.bitwise_and(qweight, mask)
    w_high = jnp.bitwise_and(lax.shift_right_logical(qweight, shift), mask)
    w_unpacked = jnp.stack([w_low, w_high], axis=1).reshape(K, N)

    w_grouped = w_unpacked.reshape(G, group_size, N).astype(x.dtype)
    w_dequant = (w_grouped - zeros[:, None, :]) * scales[:, None, :]
    w = w_dequant.reshape(K, N)

    out = jnp.dot(x, w, preferred_element_type=jnp.float32)
    return out.astype(x.dtype)
