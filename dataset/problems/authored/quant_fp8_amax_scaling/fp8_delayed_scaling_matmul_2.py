"""FP8 (e4m3) delayed-scaling matmul: amax-history scales -> quantize x,w to fp8 -> f32-accumulated matmul -> dequantize -> amax history update."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'fp8_delayed_scaling_matmul',
    'M': 8192,
    'K': 4096,
    'N': 4096,
    'amax_history_len': 16,
    'fp8_max': 448.0,
    'margin': 0,
    'history_amax_low': 1.0,
    'history_amax_high': 4.0,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kw, khx, khw = jax.random.split(key, 4)
    M, K, N, H = CONFIG['M'], CONFIG['K'], CONFIG['N'], CONFIG['amax_history_len']
    x = jax.random.normal(kx, (M, K), dtype)
    w = jax.random.normal(kw, (K, N), dtype) * CONFIG['weight_scale']
    lo, hi = CONFIG['history_amax_low'], CONFIG['history_amax_high']
    hist_x = jax.random.uniform(khx, (H,), jnp.float32, lo, hi)
    hist_w = jax.random.uniform(khw, (H,), jnp.float32, lo, hi) * CONFIG['weight_scale']
    return x, w, hist_x, hist_w


def _scale_from_history(hist):
    fp8_max = jnp.float32(CONFIG['fp8_max'])
    amax = jnp.max(hist)
    sf = (fp8_max / amax) / jnp.float32(2.0 ** CONFIG['margin'])
    sf = jnp.where(amax > 0.0, sf, jnp.float32(1.0))
    sf = jnp.where(jnp.isfinite(amax), sf, jnp.float32(1.0))
    return sf


def _quantize(t, scale):
    fp8_max = jnp.float32(CONFIG['fp8_max'])
    ts = jnp.clip(t.astype(jnp.float32) * scale, -fp8_max, fp8_max)
    return ts.astype(jnp.float8_e4m3fn)


def _update_history(hist, amax_now):
    return jnp.roll(hist, 1).at[0].set(amax_now)


def workload(x, w, hist_x, hist_w):
    sx = _scale_from_history(hist_x)
    sw = _scale_from_history(hist_w)

    xq = _quantize(x, sx)
    wq = _quantize(w, sw)

    # fp8 values are exactly representable in bf16; accumulate in float32.
    acc = lax.dot_general(
        xq.astype(jnp.bfloat16), wq.astype(jnp.bfloat16),
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    y32 = acc * (1.0 / (sx * sw))
    y = y32.astype(x.dtype)

    amax_x = jnp.max(jnp.abs(x.astype(jnp.float32)))
    amax_w = jnp.max(jnp.abs(w.astype(jnp.float32)))
    amax_y = jnp.max(jnp.abs(y32))

    new_hist_x = _update_history(hist_x, amax_x)
    new_hist_w = _update_history(hist_w, amax_w)
    return y, new_hist_x, new_hist_w, amax_y
