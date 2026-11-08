"""FP8 e4m3 delayed-scaling matmul: amax-history scale -> saturating fp8 cast -> f32 matmul -> unscale -> amax history update."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'fp8_delayed_scaling_matmul',
    'M': 8192,
    'K': 8192,
    'N': 8192,
    'amax_history_len': 16,
    'margin': 0,
    'fp8_max': 448.0,
    'hist_low': 0.5,
    'hist_high': 4.0,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    kx, kw, khx, khw = jax.random.split(key, 4)
    M, K, N = CONFIG['M'], CONFIG['K'], CONFIG['N']
    L = CONFIG['amax_history_len']
    x = jax.random.normal(kx, (M, K), dtype)
    w = jax.random.normal(kw, (K, N), dtype) * 0.02
    amax_hist_x = jax.random.uniform(
        khx, (L,), jnp.float32, CONFIG['hist_low'], CONFIG['hist_high'])
    amax_hist_w = jax.random.uniform(
        khw, (L,), jnp.float32, CONFIG['hist_low'], CONFIG['hist_high']) * 0.02
    return x, w, amax_hist_x, amax_hist_w


def _scale_from_history(hist):
    amax = jnp.max(hist)
    fp8_max = jnp.float32(CONFIG['fp8_max'])
    s = fp8_max / amax / jnp.float32(2.0 ** CONFIG['margin'])
    return jnp.where(amax > 0.0, s, jnp.float32(1.0))


def _quantize(t, scale):
    fp8_max = jnp.float32(CONFIG['fp8_max'])
    ts = jnp.clip(t.astype(jnp.float32) * scale, -fp8_max, fp8_max)
    return ts.astype(jnp.float8_e4m3fn)


def _update_history(hist, t):
    cur = jnp.max(jnp.abs(t.astype(jnp.float32)))
    return jnp.roll(hist, 1).at[0].set(cur)


def workload(x, w, amax_hist_x, amax_hist_w):
    sx = _scale_from_history(amax_hist_x)
    sw = _scale_from_history(amax_hist_w)
    xq = _quantize(x, sx)
    wq = _quantize(w, sw)
    acc = jnp.dot(xq.astype(jnp.float32), wq.astype(jnp.float32),
                  precision=lax.Precision.HIGHEST,
                  preferred_element_type=jnp.float32)
    out = (acc * (jnp.float32(1.0) / (sx * sw))).astype(x.dtype)
    new_hist_x = _update_history(amax_hist_x, x)
    new_hist_w = _update_history(amax_hist_w, w)
    return out, new_hist_x, new_hist_w
