"""Weight-only INT4 packed group-quantized matmul (W4A16) with per-group scales and zero-points."""
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
    'bit_mask': 15,
    'bit_shift': 4,
}

def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    M = CONFIG['M']
    K = CONFIG['K']
    N = CONFIG['N']
    group_size = CONFIG['group_size']
    pack_factor = CONFIG['pack_factor']
    num_groups = K // group_size

    x = jax.random.normal(k1, (M, K), dtype=dtype)
    w_packed = jax.random.randint(k2, (K // pack_factor, N), minval=-128, maxval=128, dtype=jnp.int8)
    scales = jax.random.normal(k3, (num_groups, N), dtype=dtype) * 0.02
    zeros = jax.random.uniform(k4, (num_groups, N), dtype=dtype, minval=0.0, maxval=15.0)
    return x, w_packed, scales, zeros

def workload(x, w_packed, scales, zeros):
    M = CONFIG['M']
    K = CONFIG['K']
    N = CONFIG['N']
    group_size = CONFIG['group_size']
    pack_factor = CONFIG['pack_factor']
    bit_mask = CONFIG['bit_mask']
    bit_shift = CONFIG['bit_shift']
    num_groups = K // group_size
    packed_group_size = group_size // pack_factor

    w_u8 = lax.bitcast_convert_type(w_packed, jnp.uint8)
    w_low = w_u8 & jnp.uint8(bit_mask)
    w_high = (w_u8 >> jnp.uint8(bit_shift)) & jnp.uint8(bit_mask)

    w_unpacked = jnp.stack([w_low, w_high], axis=1).reshape(num_groups, packed_group_size * pack_factor, N)
    w_dequant = (w_unpacked.astype(x.dtype) - zeros[:, None, :]) * scales[:, None, :]
    w_dequant = w_dequant.reshape(K, N)

    out = jnp.dot(x, w_dequant, preferred_element_type=jnp.float32)
    return out.astype(x.dtype)
