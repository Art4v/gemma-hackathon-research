"""DeepSeek-V3 fp8 linear: dynamic 1x128 activation fp8 quant -> 128x128 block-scaled fp8 weight dequant -> matmul (f32 acc) -> residual add."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'fp8_act_quant_linear_residual',
    'm': 8192,
    'k': 7168,
    'n': 7168,
    'act_group': 128,
    'w_block': 128,
    'fp8_max': 448.0,
    'amax_eps': 1e-12,
    'weight_std': 0.02,
}


def _quantize_weight_blockwise(w):
    n, k = w.shape
    b = CONFIG['w_block']
    wb = w.reshape(n // b, b, k // b, b)
    amax = jnp.max(jnp.abs(wb), axis=(1, 3))
    scale = jnp.maximum(amax, CONFIG['amax_eps']) / CONFIG['fp8_max']
    q = (wb / scale[:, None, :, None]).astype(jnp.float8_e4m3fn)
    return q.reshape(n, k), scale.astype(jnp.float32)


def create_inputs(dtype=jnp.float32):
    m, k, n = CONFIG['m'], CONFIG['k'], CONFIG['n']
    key = jax.random.key(0)
    kx, kr, kw = jax.random.split(key, 3)
    x = jax.random.normal(kx, (m, k), dtype)
    residual = jax.random.normal(kr, (m, n), dtype)
    w = jax.random.normal(kw, (n, k), jnp.float32) * CONFIG['weight_std']
    w_fp8, w_scale = _quantize_weight_blockwise(w)
    return x, residual, w_fp8, w_scale


def workload(x, residual, w_fp8, w_scale):
    m, k = x.shape
    n = w_fp8.shape[0]
    g = CONFIG['act_group']
    b = CONFIG['w_block']
    fp8_max = CONFIG['fp8_max']

    # Dynamic per-token, per-128-group activation quantization (DeepSeek act_quant).
    xg = x.astype(jnp.float32).reshape(m, k // g, g)
    a_amax = jnp.max(jnp.abs(xg), axis=-1, keepdims=True)
    a_scale = jnp.maximum(a_amax, CONFIG['amax_eps']) / fp8_max
    x_q = jnp.clip(xg / a_scale, -fp8_max, fp8_max).astype(jnp.float8_e4m3fn)
    x_dq = (x_q.astype(jnp.float32) * a_scale).reshape(m, k)

    # 128x128 block-scaled weight dequantization.
    wb = w_fp8.astype(jnp.float32).reshape(n // b, b, k // b, b)
    w_dq = (wb * w_scale[:, None, :, None]).reshape(n, k)

    y = lax.dot_general(
        x_dq, w_dq,
        dimension_numbers=(((1,), (1,)), ((), ())),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    out = y + residual.astype(jnp.float32)
    return out.astype(x.dtype)
