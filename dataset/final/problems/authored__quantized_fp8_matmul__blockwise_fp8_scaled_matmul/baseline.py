"""Blockwise-scaled FP8 (e4m3fn, 128x128 block f32 scales) dequantize + matmul (DeepSeek-V3 fp8 linear)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'blockwise_fp8_scaled_matmul',
    'M': 2048,          # tokens in the prefill chunk
    'K': 7168,          # DeepSeek-V3 hidden size
    'N': 18432,         # DeepSeek-V3 dense MLP intermediate size
    'block': 128,       # fp8 quantization block (block x block)
    'fp8_max': 448.0,   # float8_e4m3fn max finite value
    'act_scale': 1.0,   # std of the synthetic activations
    'weight_scale': 0.02,
}


def _block_quantize(x, block, fp8_max):
    """Quantize a 2-D f32 array to float8_e4m3fn with one f32 scale per block x block tile."""
    rows, cols = x.shape
    xb = x.reshape(rows // block, block, cols // block, block)
    amax = jnp.max(jnp.abs(xb), axis=(1, 3), keepdims=True)
    scale = jnp.maximum(amax, 1e-12) / fp8_max                       # (rows/b, 1, cols/b, 1)
    q = jnp.clip(xb / scale, -fp8_max, fp8_max).astype(jnp.float8_e4m3fn)
    return q.reshape(rows, cols), scale.reshape(rows // block, cols // block).astype(jnp.float32)


def create_inputs(dtype=jnp.float32):
    m, k, n, b = CONFIG['M'], CONFIG['K'], CONFIG['N'], CONFIG['block']
    key = jax.random.key(0)
    ka, kw = jax.random.split(key)
    a = jax.random.normal(ka, (m, k), jnp.float32) * CONFIG['act_scale']
    w = jax.random.normal(kw, (n, k), jnp.float32) * CONFIG['weight_scale']
    a_q, a_s = _block_quantize(a, b, CONFIG['fp8_max'])
    w_q, w_s = _block_quantize(w, b, CONFIG['fp8_max'])
    # a_q: (M, K) e4m3fn, a_s: (M/b, K/b) f32, w_q: (N, K) e4m3fn, w_s: (N/b, K/b) f32
    return a_q, a_s, w_q, w_s


def _block_dequantize(q, s, block):
    rows, cols = q.shape
    x = q.astype(jnp.float32).reshape(rows // block, block, cols // block, block)
    x = x * s[:, None, :, None]
    return x.reshape(rows, cols)


def workload(a_q, a_s, w_q, w_s):
    b = CONFIG['block']
    a = _block_dequantize(a_q, a_s, b)   # (M, K) f32
    w = _block_dequantize(w_q, w_s, b)   # (N, K) f32
    y = lax.dot_general(
        a, w, (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )                                    # (M, N) f32 accumulation
    return y.astype(jnp.bfloat16)
