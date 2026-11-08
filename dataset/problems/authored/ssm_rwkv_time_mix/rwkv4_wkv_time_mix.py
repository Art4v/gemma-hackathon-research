"""RWKV-4 time-mixing WKV recurrence with per-channel exponential decay and receptance gating."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'rwkv4_wkv_time_mix',
    'batch_size': 8,
    'seq_len': 2048,
    'channels': 4096,
    'eps': 1e-12,
    'init_neg_inf': -1e30,
}

def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch_size']
    T = CONFIG['seq_len']
    C = CONFIG['channels']
    k0, k1, k2, k3, k4 = jax.random.split(jax.random.key(0), 5)
    r = jax.random.normal(k0, (B, T, C), dtype=dtype)
    k = jax.random.uniform(k1, (B, T, C), dtype=dtype, minval=-2.0, maxval=2.0)
    v = jax.random.normal(k2, (B, T, C), dtype=dtype)
    w = -jax.nn.softplus(jax.random.normal(k3, (C,), dtype=jnp.float32))
    u = jax.random.normal(k4, (C,), dtype=jnp.float32) * 0.02
    return r, k, v, w, u

def workload(r, k, v, w, u):
    B, T, C = k.shape
    eps = CONFIG['eps']
    init_neg_inf = CONFIG['init_neg_inf']

    r_T = jnp.swapaxes(r, 0, 1)
    k_T = jnp.swapaxes(k, 0, 1)
    v_T = jnp.swapaxes(v, 0, 1)

    init_a = jnp.zeros((B, C), dtype=jnp.float32)
    init_b = jnp.zeros((B, C), dtype=jnp.float32)
    init_o = jnp.full((B, C), init_neg_inf, dtype=jnp.float32)

    w_f32 = w.astype(jnp.float32)[None, :]
    u_f32 = u.astype(jnp.float32)[None, :]

    def step(carry, inputs):
        a, b, o = carry
        rt, kt, vt = inputs
        rt_f32 = rt.astype(jnp.float32)
        kt_f32 = kt.astype(jnp.float32)
        vt_f32 = vt.astype(jnp.float32)

        uk = u_f32 + kt_f32
        p = jnp.maximum(o, uk)
        e1 = jnp.exp(o - p)
        e2 = jnp.exp(uk - p)
        wkv = (e1 * a + e2 * vt_f32) / (e1 * b + e2 + eps)
        out_t = (jax.nn.sigmoid(rt_f32) * wkv).astype(r.dtype)

        wo = o + w_f32
        q = jnp.maximum(wo, kt_f32)
        e1_next = jnp.exp(wo - q)
        e2_next = jnp.exp(kt_f32 - q)
        a_next = e1_next * a + e2_next * vt_f32
        b_next = e1_next * b + e2_next
        o_next = q

        return (a_next, b_next, o_next), out_t

    (_, _, _), out_T = lax.scan(step, (init_a, init_b, init_o), (r_T, k_T, v_T))
    return jnp.swapaxes(out_T, 0, 1)
