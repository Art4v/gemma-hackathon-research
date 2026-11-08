"""Mamba-1 selective scan with softplus discretization, float32 state recurrence, skip connection, and SiLU gating."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'mamba1_selective_scan',
    'batch_size': 8,
    'seq_len': 2048,
    'd_inner': 2048,
    'd_state': 16,
}


def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch_size']
    L = CONFIG['seq_len']
    D = CONFIG['d_inner']
    N = CONFIG['d_state']

    k_u, k_dt, k_B, k_C, k_z, k_A, k_b, k_D = jax.random.split(jax.random.key(0), 8)

    u = jax.random.normal(k_u, (B, L, D), dtype=dtype)
    delta = jax.random.normal(k_dt, (B, L, D), dtype=dtype) - jnp.array(2.0, dtype=dtype)
    B_proj = jax.random.normal(k_B, (B, L, N), dtype=dtype) * jnp.array(0.1, dtype=dtype)
    C_proj = jax.random.normal(k_C, (B, L, N), dtype=dtype) * jnp.array(0.1, dtype=dtype)
    z = jax.random.normal(k_z, (B, L, D), dtype=dtype)

    A = -jax.random.uniform(k_A, (D, N), dtype=jnp.float32, minval=1.0, maxval=16.0)
    dt_bias = jax.random.uniform(k_b, (D,), dtype=jnp.float32, minval=-4.0, maxval=-1.0)
    D_weight = jnp.ones((D,), dtype=jnp.float32)

    return u, delta, B_proj, C_proj, z, A, dt_bias, D_weight


def workload(u, delta, B_proj, C_proj, z, A, dt_bias, D_weight):
    out_dtype = u.dtype
    B, L, D = u.shape
    N = A.shape[1]

    u_f = u.astype(jnp.float32)
    delta_f = delta.astype(jnp.float32)
    B_f = B_proj.astype(jnp.float32)
    C_f = C_proj.astype(jnp.float32)
    z_f = z.astype(jnp.float32)
    A_f = A.astype(jnp.float32)
    dt_bias_f = dt_bias.astype(jnp.float32)
    D_f = D_weight.astype(jnp.float32)

    dt = jax.nn.softplus(delta_f + dt_bias_f[None, None, :])

    u_t = jnp.swapaxes(u_f, 0, 1)
    dt_t = jnp.swapaxes(dt, 0, 1)
    B_t = jnp.swapaxes(B_f, 0, 1)
    C_t = jnp.swapaxes(C_f, 0, 1)

    def step(h, xs):
        u_i, dt_i, B_i, C_i = xs
        dA = jnp.exp(dt_i[:, :, None] * A_f[None, :, :])
        dB_u = (dt_i * u_i)[:, :, None] * B_i[:, None, :]
        h_new = dA * h + dB_u
        y_i = jnp.einsum('bdn,bn->bd', h_new, C_i)
        return h_new, y_i

    h0 = jnp.zeros((B, D, N), dtype=jnp.float32)
    _, y_t = lax.scan(step, h0, (u_t, dt_t, B_t, C_t))
    y = jnp.swapaxes(y_t, 0, 1)

    y = y + u_f * D_f[None, None, :]
    out = y * jax.nn.silu(z_f)
    return out.astype(out_dtype)
