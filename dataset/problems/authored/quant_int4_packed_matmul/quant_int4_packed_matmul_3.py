"""Weight-only INT4 packed group-quantized matrix multiplication (W4A16 AWQ/GPTQ linear)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'quant_int4_packed_matmul',
    'M': 2048,
    'K': 8192,
    'N': 28672,
    'group_size': 128,
    'pack_factor': 2,
    'bit_width': 4,
    'mask': 15,
}


def create_inputs(dtype=jnp.bfloat16):
    k0, k1, k2, k3 = jax.random.split(jax.random.key(0), 4)
    M = CONFIG['M']
    K = CONFIG['K']
    N = CONFIG['N']
    group_size = CONFIG['group_size']
    pack_factor = CONFIG['pack_factor']
    num_groups = K // group_size

    x = jax.random.normal(k0, (M, K), dtype=dtype)
    qweight_packed = jax.random.randint(
        k1, (K // pack_factor, N), minval=0, maxval=256, dtype=jnp.uint8
    )
    scales = jax.random.normal(k2, (num_groups, N), dtype=dtype) * 0.02
    zeros = jax.random.uniform(k3, (num_groups, N), dtype=dtype, minval=6.0, maxval=10.0)
    return x, qweight_packed, scales, zeros


def workload(x, qweight_packed, scales, zeros):
    K = CONFIG['K']
    N = CONFIG['N']
    group_size = CONFIG['group_size']
    pack_factor = CONFIG['pack_factor']
    bit_width = CONFIG['bit_width']
    mask = CONFIG['mask']
    num_groups = K // group_size
    packed_group_size = group_size // pack_factor

    qw_grouped = qweight_packed.reshape(num_groups, packed_group_size, N)
    w_low = jnp.bitwise_and(qw_grouped, mask)
    w_high = jnp.bitwise_and(jnp.right_shift(qw_grouped, bit_width), mask)

    w_unpacked = jnp.stack([w_low, w_high], axis=2).reshape(num_groups, group_size, N)
    w_dequant = (w_unpacked.astype(x.dtype) - zeros[:, None, :]) * scales[:, None, :]
    w_dequant = w_dequant.reshape(K, N)

    out = jnp.matmul(x, w_dequant, preferred_element_type=jnp.float32)
    return out.astype(x.dtype)
