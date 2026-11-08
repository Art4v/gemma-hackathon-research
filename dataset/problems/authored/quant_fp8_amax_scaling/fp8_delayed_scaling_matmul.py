"""FP8 (e4m3) delayed-scaling matmul: amax-history scale -> quantize -> f32 matmul -> unscale, plus amax history update."""
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
    'x_amax_hist_low': 3.0,
    'x_amax_hist_high': 6.0,
    'w_amax_hist_low': 0.07,
    'w_amax_hist_high': 0.12,
    'weight_std': 0.02,
}


def create_inputs(dtype=jnp.float32):
    key = jax.random.key(0)
    kx, kw, khx, khw = jax.random.split(key, 4)
    M, K, N, H = CONFIG['M'], CONFIG['K'], CONFIG['N'], CONFIG['amax_history_len']
    x = jax.random.normal(kx, (M, K), dtype)
    w = jax.random.normal(kw, (K, N), dtype) * CONFIG['weight_std']
    hist_x = jax.random.uniform(khx, (H,), jnp.float32,
                                CONFIG['x_amax_hist_low'], CONFIG['x_amax_hist_high'])
    hist_w = jax.random.uniform(khw, (H,), jnp.float32,
                                CONFIG['w_amax_hist_low'], CONFIG['w_amax_hist_high'])
    return x, w, hist_x, hist_w


def _scale_from_history(hist):
    fp8_max = CONFIG['fp8_max']
    amax = jnp.max(hist).astype(jnp.float32)
    scale = fp8_max / amax / (2.0 ** CONFIG['margin'])
    # Guard against zero / non-finite amax (as in Transformer Engine).
    scale = jnp.where(jnp.isfinite(amax) & (amax > 0.0), scale, jnp.float32(1.0))
    return scale


def _quantize_e4m3(t, scale):
    fp8_max = CONFIG['fp8_max']
    ts = t.astype(jnp.float32) * scale
    ts = jnp.clip(ts, -fp8_max, fp8_max)
    return ts.astype(jnp.float8_e4m3fn)


def _update_history(hist, t):
    cur_amax = jnp.max(jnp.abs(t.astype(jnp.float32)))
    rolled = jnp.roll(hist, 1)
    return rolled.at[0].set(cur_amax)


def workload(x, w, hist_x, hist_w):
    scale_x = _scale_from_history(hist_x)
    scale_w = _scale_from_history(hist_w)

    x_fp8 = _quantize_e4m3(x, scale_x)
    w_fp8 = _quantize_e4m3(w, scale_w)

    acc = lax.dot_general(
        x_fp8.astype(jnp.float32), w_fp8.astype(jnp.float32),
        dimension_numbers=(((1,), (0,)), ((), ())),
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    inv_scale = 1.0 / (scale_x * scale_w)
    y = (acc * inv_scale).astype(x.dtype)

    new_hist_x = _update_history(hist_x, x)
    new_hist_w = _update_history(hist_w, w)
    return y, new_hist_x, new_hist_w
