"""DeepSeek-V3 blockwise-scaled FP8 (float8_e4m3fn) matrix multiplication with 1x128 activation and 128x128 weight scales."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'deepseek_v3_blockwise_fp8_gemm',
    'M': 4096,
    'K': 7168,
    'N': 18432,
    'block_size': 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    M = CONFIG['M']
    K = CONFIG['K']
    N = CONFIG['N']
    bs = CONFIG['block_size']

    x_fp8 = jax.random.normal(k1, (M, K), dtype=jnp.float32).astype(jnp.float8_e4m3fn)
    x_scale = jax.random.uniform(k2, (M, K // bs), dtype=jnp.float32, minval=0.001, maxval=0.05)

    w_fp8 = (jax.random.normal(k3, (K, N), dtype=jnp.float32) * 0.02).astype(jnp.float8_e4m3fn)
    w_scale = jax.random.uniform(k4, (K // bs, N // bs), dtype=jnp.float32, minval=0.001, maxval=0.05)

    return x_fp8, x_scale, w_fp8, w_scale


def workload(x_fp8, x_scale, w_fp8, w_scale):
    M, K = x_fp8.shape
    _, N = w_fp8.shape
    bs = CONFIG['block_size']

    x_reshaped = x_fp8.astype(jnp.float32).reshape(M, K // bs, bs)
    x_dequant = (x_reshaped * x_scale[:, :, None]).reshape(M, K)

    w_reshaped = w_fp8.astype(jnp.float32).reshape(K // bs, bs, N // bs, bs)
    w_dequant = (w_reshaped * w_scale[:, None, :, None]).reshape(K, N)

    out = jnp.dot(x_dequant, w_dequant, preferred_element_type=jnp.float32)
    return out.astype(jnp.bfloat16)
