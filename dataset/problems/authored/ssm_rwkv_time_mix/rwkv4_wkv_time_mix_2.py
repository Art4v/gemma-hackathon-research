"""RWKV-4 time-mixing WKV recurrence with max-stabilized f32 state and sigmoid receptance gating."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'rwkv4_wkv_time_mix',
    'batch_size': 8,
    'seq_len': 2048,
    'channels': 4096,
    'eps': 1e-12,
}

def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch_size']
    T = CONFIG['seq_len']
    C = CONFIG['channels']
    key = jax.random.key(0)
    k_r, k_k, k_v, k_w, k_u, k_a, k_b, k_p = jax.random.split(key, 8)

    r = jax.random.normal(k_r, (B, T, C), dtype=dtype)
    k = jax.random.uniform(k_k, (B, T, C), dtype=dtype, minval=-2.0, maxval=2.0)
    v = jax.random.normal(k_v, (B, T, C), dtype=dtype)

    w = -jax.random.uniform(k_w, (C,), dtype=jnp.float32, minval=0.05, maxval=1.5)
    u = jax.random.normal(k_u, (C,), dtype=jnp.float32) * 0.02

    state_a = jax.random.normal(k_a, (B, C), dtype=jnp.float32) * 0.02
    state_b = jax.random.uniform(k_b, (B, C), dtype=jnp.float32, minval=0.1, maxval=1.0)
    state_p = jax.random.uniform(k_p, (B, C), dtype=jnp.float32, minval=-2.0, maxval=0.0)

    return r, k, v, w, u, state_a, state_b, state_p

def workload(r, k, v, w, u, state_a, state_b, state_p):
    orig_dtype = r.dtype
    eps = CONFIG['eps']

    r_t = jnp.swapaxes(r.astype(jnp.float32), 0, 1)
    k_t = jnp.swapaxes(k.astype(jnp.float32), 0, 1)
    v_t = jnp.swapaxes(v.astype(jnp.float32), 0, 1)

    w_f32 = -jnp.exp(w.astype(jnp.float32)) if False else w.astype(jnp.float32)
    u_f32 = u.astype(jnp.float32)

    init_carry = (
        state_a.astype(jnp.float32),
        state_b.astype(jnp.float32),
        state_p.astype(jnp.float32),
    )

    def step_fn(carry, inputs):
        aa, bb, pp = carry
        rt, kt, vt = inputs

        ww = u_f32 + kt
        q = jnp.maximum(pp, ww)
        e1 = jnp.exp(pp - q)
        e2 = jnp.exp(ww - q)

        wkv = (e1 * aa + e2 * vt) / (e1 * bb + e2 + eps)
        rwkv = jax.nn.sigmoid(rt) * wkv

        ww_next = pp + w_f32
        q_next = jnp.maximum(ww_next, kt)
        e1_next = jnp.exp(ww_next - q_next)
        e2_next = jnp.exp(kt - q_next)

        aa_next = e1_next * aa + e2_next * vt
        bb_next = e1_next * bb + e2_next
        pp_next = q_next

        return (aa_next, bb_next, pp_next), rwkv.astype(orig_dtype)

    final_carry, out_t = lax.scan(step_fn, init_carry, (r_t, k_t, v_t))
    out = jnp.swapaxes(out_t, 0, 1)
    final_a, final_b, final_p = final_carry
    return out, final_a, final_b, final_p
