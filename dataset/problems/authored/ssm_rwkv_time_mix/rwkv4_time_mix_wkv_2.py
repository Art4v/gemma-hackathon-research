"""RWKV-4 time-mix: token shift -> K/V/R projections -> WKV exponential-decay recurrence (f32 state) -> sigmoid(R) gate -> output projection."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'rwkv4_time_mix_wkv',
    'batch': 8,
    'seq_len': 1024,
    'd_model': 2048,
    'weight_scale': 0.02,
    'decay_min': -5.0,
    'decay_max': 3.0,
    'decay_power_base': 0.7,
    'first_base': 0.3,
    'init_pp': -1e38,
}


def create_inputs(dtype=jnp.float32):
    B, T, C = CONFIG['batch'], CONFIG['seq_len'], CONFIG['d_model']
    s = CONFIG['weight_scale']
    key = jax.random.key(0)
    kx, kk, kv, kr, ko = jax.random.split(key, 5)
    x = jax.random.normal(kx, (B, T, C), dtype)
    w_k = jax.random.normal(kk, (C, C), dtype) * s
    w_v = jax.random.normal(kv, (C, C), dtype) * s
    w_r = jax.random.normal(kr, (C, C), dtype) * s
    w_o = jax.random.normal(ko, (C, C), dtype) * s
    # Deterministic RWKV-style per-channel initializations.
    ratio = jnp.arange(C, dtype=jnp.float32) / (C - 1)
    mix_k = ratio.astype(dtype)
    mix_v = (ratio + 0.3 * 0.5).clip(0.0, 1.0).astype(dtype)
    mix_r = (0.5 * ratio).astype(dtype)
    d_lo, d_hi = CONFIG['decay_min'], CONFIG['decay_max']
    time_decay = d_lo + (d_hi - d_lo) * ratio ** (CONFIG['decay_power_base'] + 1.3 * ratio)
    zigzag = ((jnp.arange(C) + 1) % 3 - 1).astype(jnp.float32) * 0.5
    time_first = jnp.full((C,), jnp.log(CONFIG['first_base']), jnp.float32) + zigzag
    return (x, mix_k, mix_v, mix_r, time_decay, time_first, w_k, w_v, w_r, w_o)


def _wkv(w, u, k, v, init_pp):
    # k, v: (T, B, C) float32; w, u: (C,) float32
    B, C = k.shape[1], k.shape[2]
    aa0 = jnp.zeros((B, C), jnp.float32)
    bb0 = jnp.zeros((B, C), jnp.float32)
    pp0 = jnp.full((B, C), init_pp, jnp.float32)

    def step(carry, kv_t):
        aa, bb, pp = carry
        kt, vt = kv_t
        ww = u + kt
        p = jnp.maximum(pp, ww)
        e1 = jnp.exp(pp - p)
        e2 = jnp.exp(ww - p)
        out = (e1 * aa + e2 * vt) / (e1 * bb + e2)
        ww = w + pp
        p = jnp.maximum(ww, kt)
        e1 = jnp.exp(ww - p)
        e2 = jnp.exp(kt - p)
        aa = e1 * aa + e2 * vt
        bb = e1 * bb + e2
        return (aa, bb, p), out

    _, outs = lax.scan(step, (aa0, bb0, pp0), (k, v))
    return outs  # (T, B, C)


def workload(x, mix_k, mix_v, mix_r, time_decay, time_first, w_k, w_v, w_r, w_o):
    dtype = x.dtype
    # Token shift: previous token (zero for first position).
    xx = jnp.concatenate([jnp.zeros_like(x[:, :1, :]), x[:, :-1, :]], axis=1)
    xk = x * mix_k + xx * (1 - mix_k)
    xv = x * mix_v + xx * (1 - mix_v)
    xr = x * mix_r + xx * (1 - mix_r)

    k = jnp.einsum('btc,cd->btd', xk, w_k, preferred_element_type=jnp.float32)
    v = jnp.einsum('btc,cd->btd', xv, w_v, preferred_element_type=jnp.float32)
    r = jnp.einsum('btc,cd->btd', xr, w_r, preferred_element_type=jnp.float32)

    w = -jnp.exp(time_decay.astype(jnp.float32))
    u = time_first.astype(jnp.float32)
    wkv = _wkv(w, u, jnp.swapaxes(k, 0, 1), jnp.swapaxes(v, 0, 1), CONFIG['init_pp'])
    wkv = jnp.swapaxes(wkv, 0, 1)  # (B, T, C)

    gated = (jax.nn.sigmoid(r) * wkv).astype(dtype)
    out = jnp.einsum('btc,cd->btd', gated, w_o, preferred_element_type=jnp.float32)
    return out.astype(dtype)
