"""RWKV-4 time-mix: token shift -> r/k/v projections -> stable WKV exponential-decay recurrence (f32 state) -> sigmoid(r) gating -> output projection."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'rwkv4_time_mix_wkv',
    'batch': 8,
    'seq_len': 2048,
    'd_model': 2048,
    'weight_scale': 0.02,
    'decay_min': -5.0,
    'decay_max': 1.0,
    'first_bonus': 0.5,
    'init_pp': -1e38,
}


def create_inputs(dtype=jnp.bfloat16):
    B, T, C = CONFIG['batch'], CONFIG['seq_len'], CONFIG['d_model']
    s = CONFIG['weight_scale']
    key = jax.random.key(0)
    kx, kk, kv, kr, ko = jax.random.split(key, 5)
    x = jax.random.normal(kx, (B, T, C), dtype)
    w_k = jax.random.normal(kk, (C, C), dtype) * s
    w_v = jax.random.normal(kv, (C, C), dtype) * s
    w_r = jax.random.normal(kr, (C, C), dtype) * s
    w_o = jax.random.normal(ko, (C, C), dtype) * s
    ratio = jnp.linspace(0.0, 1.0, C, dtype=jnp.float32)
    time_mix_k = ratio.astype(dtype)
    time_mix_v = (ratio * 0.7 + 0.3 * ratio ** 0.5).astype(dtype)
    time_mix_r = (ratio ** 0.5).astype(dtype)
    time_decay = jnp.linspace(CONFIG['decay_min'], CONFIG['decay_max'], C,
                              dtype=jnp.float32)
    zigzag = (jnp.arange(C) % 3 - 1).astype(jnp.float32) * 0.5
    time_first = jnp.full((C,), jnp.log(0.3), jnp.float32) + CONFIG['first_bonus'] * zigzag
    return (x, w_k, w_v, w_r, w_o, time_mix_k, time_mix_v, time_mix_r,
            time_decay, time_first)


def _wkv(w, u, k, v):
    # k, v: (B, T, C) float32; w, u: (C,) float32. Stable RWKV-4 recurrence.
    B, T, C = k.shape
    k_t = jnp.swapaxes(k, 0, 1)
    v_t = jnp.swapaxes(v, 0, 1)

    def step(carry, kv):
        aa, bb, pp = carry
        kk, vv = kv
        ww = u + kk
        p = jnp.maximum(pp, ww)
        e1 = jnp.exp(pp - p)
        e2 = jnp.exp(ww - p)
        out = (e1 * aa + e2 * vv) / (e1 * bb + e2)
        ww = pp + w
        p = jnp.maximum(ww, kk)
        e1 = jnp.exp(ww - p)
        e2 = jnp.exp(kk - p)
        aa = e1 * aa + e2 * vv
        bb = e1 * bb + e2
        return (aa, bb, p), out

    init = (jnp.zeros((B, C), jnp.float32),
            jnp.zeros((B, C), jnp.float32),
            jnp.full((B, C), CONFIG['init_pp'], jnp.float32))
    _, out = lax.scan(step, init, (k_t, v_t))
    return jnp.swapaxes(out, 0, 1)


def workload(x, w_k, w_v, w_r, w_o, time_mix_k, time_mix_v, time_mix_r,
             time_decay, time_first):
    xf = x.astype(jnp.float32)
    xx = jnp.pad(xf, ((0, 0), (1, 0), (0, 0)))[:, :-1, :]
    mk = time_mix_k.astype(jnp.float32)
    mv = time_mix_v.astype(jnp.float32)
    mr = time_mix_r.astype(jnp.float32)
    xk = (xf * mk + xx * (1.0 - mk)).astype(x.dtype)
    xv = (xf * mv + xx * (1.0 - mv)).astype(x.dtype)
    xr = (xf * mr + xx * (1.0 - mr)).astype(x.dtype)

    k = jnp.matmul(xk, w_k, preferred_element_type=jnp.float32)
    v = jnp.matmul(xv, w_v, preferred_element_type=jnp.float32)
    r = jax.nn.sigmoid(jnp.matmul(xr, w_r, preferred_element_type=jnp.float32))

    w = -jnp.exp(time_decay.astype(jnp.float32))
    u = time_first.astype(jnp.float32)
    wkv = _wkv(w, u, k, v)

    gated = (r * wkv).astype(x.dtype)
    out = jnp.matmul(gated, w_o, preferred_element_type=jnp.float32)
    return out.astype(x.dtype)
