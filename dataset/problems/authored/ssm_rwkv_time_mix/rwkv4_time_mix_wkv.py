"""RWKV-4 time mixing: token shift -> R/K/V projections -> WKV exponential-decay recurrence (f32 state) -> sigmoid(r) gating -> output projection."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'rwkv4_time_mix_wkv',
    'batch': 8,
    'seq_len': 2048,
    'd_model': 2048,
    'weight_scale': 0.02,
    'decay_lo': -5.0,
    'decay_span': 8.0,
    'decay_pow_base': 0.7,
    'decay_pow_scale': 1.3,
    'layer_ratio': 0.5,
    'first_base': 0.3,
    'first_zigzag': 0.5,
    'p_init': -1e38,
}


def create_inputs(dtype=jnp.bfloat16):
    B, T, C = CONFIG['batch'], CONFIG['seq_len'], CONFIG['d_model']
    key = jax.random.key(0)
    kx, kk, kv, kr, ko = jax.random.split(key, 5)
    x = jax.random.normal(kx, (B, T, C), dtype=jnp.float32).astype(dtype)
    s = CONFIG['weight_scale']
    w_k = (jax.random.normal(kk, (C, C), dtype=jnp.float32) * s).astype(dtype)
    w_v = (jax.random.normal(kv, (C, C), dtype=jnp.float32) * s).astype(dtype)
    w_r = (jax.random.normal(kr, (C, C), dtype=jnp.float32) * s).astype(dtype)
    w_o = (jax.random.normal(ko, (C, C), dtype=jnp.float32) * s).astype(dtype)

    ratio = CONFIG['layer_ratio']
    ramp = jnp.arange(C, dtype=jnp.float32) / jnp.float32(C)
    mix_k = jnp.power(ramp, 1.0 - ratio)
    mix_v = jnp.power(ramp, 1.0 - ratio) + 0.3 * ratio
    mix_r = jnp.power(ramp, 0.5 * (1.0 - ratio))

    idx = jnp.arange(C, dtype=jnp.float32) / jnp.float32(C - 1)
    time_decay = CONFIG['decay_lo'] + CONFIG['decay_span'] * jnp.power(
        idx, CONFIG['decay_pow_base'] + CONFIG['decay_pow_scale'] * ratio)
    zigzag = (jnp.mod(jnp.arange(C) + 1, 3).astype(jnp.float32) - 1.0) * CONFIG['first_zigzag']
    time_first = jnp.log(jnp.float32(CONFIG['first_base'])) + zigzag

    return (x, mix_k, mix_v, mix_r, time_decay, time_first, w_k, w_v, w_r, w_o)


def _wkv(w, u, k, v):
    # k, v: (B, T, C) float32; w, u: (C,) float32. Returns (B, T, C) float32.
    B, _, C = k.shape
    kt = jnp.swapaxes(k, 0, 1)
    vt = jnp.swapaxes(v, 0, 1)
    a0 = jnp.zeros((B, C), jnp.float32)
    b0 = jnp.zeros((B, C), jnp.float32)
    p0 = jnp.full((B, C), CONFIG['p_init'], jnp.float32)

    def step(carry, kv_t):
        a, b, p = carry
        k_t, v_t = kv_t
        ww = u + k_t
        q = jnp.maximum(p, ww)
        e1 = jnp.exp(p - q)
        e2 = jnp.exp(ww - q)
        y = (e1 * a + e2 * v_t) / (e1 * b + e2)
        ww2 = p + w
        q2 = jnp.maximum(ww2, k_t)
        e1 = jnp.exp(ww2 - q2)
        e2 = jnp.exp(k_t - q2)
        a = e1 * a + e2 * v_t
        b = e1 * b + e2
        return (a, b, q2), y

    _, ys = lax.scan(step, (a0, b0, p0), (kt, vt))
    return jnp.swapaxes(ys, 0, 1)


def workload(x, mix_k, mix_v, mix_r, time_decay, time_first, w_k, w_v, w_r, w_o):
    dt = x.dtype
    xf = x.astype(jnp.float32)
    xx = jnp.pad(xf, ((0, 0), (1, 0), (0, 0)))[:, :-1, :]
    xk = (xf * mix_k + xx * (1.0 - mix_k)).astype(dt)
    xv = (xf * mix_v + xx * (1.0 - mix_v)).astype(dt)
    xr = (xf * mix_r + xx * (1.0 - mix_r)).astype(dt)

    k = jnp.einsum('btc,cd->btd', xk, w_k, preferred_element_type=jnp.float32)
    v = jnp.einsum('btc,cd->btd', xv, w_v, preferred_element_type=jnp.float32)
    r = jnp.einsum('btc,cd->btd', xr, w_r, preferred_element_type=jnp.float32)

    w = -jnp.exp(time_decay.astype(jnp.float32))
    u = time_first.astype(jnp.float32)
    wkv = _wkv(w, u, k, v)

    rwkv = (jax.nn.sigmoid(r) * wkv).astype(dt)
    out = jnp.einsum('btc,cd->btd', rwkv, w_o, preferred_element_type=jnp.float32)
    return out.astype(dt)
