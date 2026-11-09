"""[m1 M3_ragged variant of authored/quantized_fp8_matmul/fp8_dynamic_act_quant_block_weight_matmul] Dynamic 1x128 fp8 activation quant fused with 128x128 block-scaled fp8 weight matmul, with a ragged token count M=4050 (not a multiple of 8 or 128)."""
import jax
import jax.numpy as jnp

CONFIG = {
    'name': 'fp8_dynamic_act_quant_block_weight_matmul_m1_ragged',
    'M': 4050,  # tokens (ragged: not a multiple of 8 or 128)
    'K': 7168,
    'N': 4608,
    'act_group': 128,
    'w_block_k': 128,
    'w_block_n': 128,
    'fp8_max': 448.0,
    'scale_eps': 1e-4,
    'weight_std': 0.02,
}


def _quantize_weight_blockwise(w):
    K, N = w.shape
    bk, bn = CONFIG['w_block_k'], CONFIG['w_block_n']
    fmax = CONFIG['fp8_max']
    wb = w.reshape(K // bk, bk, N // bn, bn)
    amax = jnp.max(jnp.abs(wb), axis=(1, 3))
    scale = jnp.maximum(amax, CONFIG['scale_eps']) / fmax
    q = jnp.clip(wb / scale[:, None, :, None], -fmax, fmax)
    wq = q.reshape(K, N).astype(jnp.float8_e4m3fn)
    return wq, scale.astype(jnp.float32)


def create_inputs(dtype=jnp.bfloat16):
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    key = jax.random.key(0)
    kx, kw = jax.random.split(key)
    x = jax.random.normal(kx, (M, K), dtype=jnp.float32).astype(dtype)
    w = jax.random.normal(kw, (K, N), dtype=jnp.float32) * CONFIG['weight_std']
    wq, w_scale = _quantize_weight_blockwise(w)
    return x, wq, w_scale


def workload(x, wq, w_scale):
    M, K = x.shape
    N = wq.shape[1]
    g = CONFIG['act_group']
    bk, bn = CONFIG['w_block_k'], CONFIG['w_block_n']
    fmax = CONFIG['fp8_max']

    # Dynamic per-token, per-128-group activation quantization (f32 math).
    xg = x.astype(jnp.float32).reshape(M, K // g, g)
    amax = jnp.max(jnp.abs(xg), axis=-1)
    x_scale = jnp.maximum(amax, CONFIG['scale_eps']) / fmax
    xq = jnp.clip(xg / x_scale[:, :, None], -fmax, fmax).astype(jnp.float8_e4m3fn)

    # Dequantize both operands to f32 (equivalent to per-block scaled accumulation).
    x_dq = (xq.astype(jnp.float32) * x_scale[:, :, None]).reshape(M, K)
    wb = wq.astype(jnp.float32).reshape(K // bk, bk, N // bn, bn)
    w_dq = (wb * w_scale[:, None, :, None]).reshape(K, N)

    out = jnp.matmul(x_dq, w_dq, preferred_element_type=jnp.float32)
    return out.astype(x.dtype)
