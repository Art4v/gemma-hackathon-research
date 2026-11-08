"""Vision Mamba (ViM) bidirectional selective scan with fused discretization and SiLU gating."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'vim_bidirectional_selective_scan',
    'batch': 8,
    'seq_len': 1024,
    'd_inner': 2048,
    'd_state': 16,
}


def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch']
    L = CONFIG['seq_len']
    D = CONFIG['d_inner']
    N = CONFIG['d_state']

    key = jax.random.key(0)
    k_u, k_z, k_df, k_db, k_bf, k_cf, k_bb, k_cb, k_af, k_ab, k_dbf, k_dbb = jax.random.split(key, 12)

    u = jax.random.normal(k_u, (B, L, D), dtype=dtype)
    z = jax.random.normal(k_z, (B, L, D), dtype=dtype)
    delta_f = jax.random.normal(k_df, (B, L, D), dtype=dtype) * 0.5 - 2.0
    delta_b = jax.random.normal(k_db, (B, L, D), dtype=dtype) * 0.5 - 2.0
    B_f = jax.random.normal(k_bf, (B, L, N), dtype=dtype) * 0.1
    C_f = jax.random.normal(k_cf, (B, L, N), dtype=dtype) * 0.1
    B_b = jax.random.normal(k_bb, (B, L, N), dtype=dtype) * 0.1
    C_b = jax.random.normal(k_cb, (B, L, N), dtype=dtype) * 0.1

    A_f = -jnp.exp(jax.random.uniform(k_af, (D, N), dtype=jnp.float32, minval=0.0, maxval=2.0))
    A_b = -jnp.exp(jax.random.uniform(k_ab, (D, N), dtype=jnp.float32, minval=0.0, maxval=2.0))
    D_f = jnp.ones((D,), dtype=jnp.float32)
    D_b = jnp.ones((D,), dtype=jnp.float32)
    delta_bias_f = jax.random.uniform(k_dbf, (D,), dtype=jnp.float32, minval=-1.0, maxval=1.0)
    delta_bias_b = jax.random.uniform(k_dbb, (D,), dtype=jnp.float32, minval=-1.0, maxval=1.0)

    return (
        u, z, delta_f, delta_b, B_f, C_f, B_b, C_b,
        A_f, A_b, D_f, D_b, delta_bias_f, delta_bias_b
    )


def workload(
    u, z, delta_f, delta_b, B_f, C_f, B_b, C_b,
    A_f, A_b, D_f, D_b, delta_bias_f, delta_bias_b
):
    orig_dtype = u.dtype
    B_sz, L, D = u.shape
    N = A_f.shape[1]

    u_f32 = u.astype(jnp.float32)
    z_f32 = z.astype(jnp.float32)

    dt_f = jax.nn.softplus(delta_f.astype(jnp.float32) + delta_bias_f[None, None, :])
    dt_b = jax.nn.softplus(delta_b.astype(jnp.float32) + delta_bias_b[None, None, :])

    B_f_f32 = B_f.astype(jnp.float32)
    C_f_f32 = C_f.astype(jnp.float32)
    B_b_f32 = B_b.astype(jnp.float32)
    C_b_f32 = C_b.astype(jnp.float32)

    u_T = jnp.transpose(u_f32, (1, 0, 2))
    dt_f_T = jnp.transpose(dt_f, (1, 0, 2))
    dt_b_T = jnp.transpose(dt_b, (1, 0, 2))
    B_f_T = jnp.transpose(B_f_f32, (1, 0, 2))
    C_f_T = jnp.transpose(C_f_f32, (1, 0, 2))
    B_b_T = jnp.transpose(B_b_f32, (1, 0, 2))
    C_b_T = jnp.transpose(C_b_f32, (1, 0, 2))

    init_h = jnp.zeros((B_sz, D, N), dtype=jnp.float32)

    def step_f(h, xs):
        u_t, dt_t, B_t, C_t = xs
        dA = jnp.exp(dt_t[:, :, None] * A_f[None, :, :])
        dB_u = (dt_t * u_t)[:, :, None] * B_t[:, None, :]
        h_next = h * dA + dB_u
        y_t = jnp.einsum('bdn,bn->bd', h_next, C_t) + D_f[None, :] * u_t
        return h_next, y_t

    _, y_f_T = lax.scan(step_f, init_h, (u_T, dt_f_T, B_f_T, C_f_T))

    def step_b(h, xs):
        u_t, dt_t, B_t, C_t = xs
        dA = jnp.exp(dt_t[:, :, None] * A_b[None, :, :])
        dB_u = (dt_t * u_t)[:, :, None] * B_t[:, None, :]
        h_next = h * dA + dB_u
        y_t = jnp.einsum('bdn,bn->bd', h_next, C_t) + D_b[None, :] * u_t
        return h_next, y_t

    _, y_b_T = lax.scan(step_b, init_h, (u_T, dt_b_T, B_b_T, C_b_T), reverse=True)

    y_T = y_f_T + y_b_T
    y = jnp.transpose(y_T, (1, 0, 2))

    out = y * jax.nn.silu(z_f32)
    return out.astype(orig_dtype)
