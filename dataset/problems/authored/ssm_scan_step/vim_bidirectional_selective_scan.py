"""Bidirectional Mamba-1 selective scan with fused delta softplus and SiLU gating (Vision Mamba / Caduceus)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'vim_bidirectional_selective_scan',
    'batch_size': 8,
    'seq_len': 2048,
    'd_inner': 1536,
    'd_state': 16,
}


def create_inputs(dtype=jnp.bfloat16):
    B = CONFIG['batch_size']
    L = CONFIG['seq_len']
    D = CONFIG['d_inner']
    N = CONFIG['d_state']

    keys = jax.random.split(jax.random.key(0), 14)

    u = jax.random.normal(keys[0], (B, L, D), dtype=dtype)
    z = jax.random.normal(keys[1], (B, L, D), dtype=dtype)

    delta_f = jax.random.normal(keys[2], (B, L, D), dtype=dtype) - 2.0
    B_f = jax.random.normal(keys[3], (B, L, N), dtype=dtype) * 0.1
    C_f = jax.random.normal(keys[4], (B, L, N), dtype=dtype) * 0.1

    delta_b = jax.random.normal(keys[5], (B, L, D), dtype=dtype) - 2.0
    B_b = jax.random.normal(keys[6], (B, L, N), dtype=dtype) * 0.1
    C_b = jax.random.normal(keys[7], (B, L, N), dtype=dtype) * 0.1

    A_f = -jnp.exp(jax.random.uniform(keys[8], (D, N), dtype=jnp.float32, minval=0.0, maxval=2.0))
    D_f = jax.random.normal(keys[9], (D,), dtype=jnp.float32) * 0.02
    delta_bias_f = jax.random.uniform(keys[10], (D,), dtype=jnp.float32, minval=-1.0, maxval=1.0)

    A_b = -jnp.exp(jax.random.uniform(keys[11], (D, N), dtype=jnp.float32, minval=0.0, maxval=2.0))
    D_b = jax.random.normal(keys[12], (D,), dtype=jnp.float32) * 0.02
    delta_bias_b = jax.random.uniform(keys[13], (D,), dtype=jnp.float32, minval=-1.0, maxval=1.0)

    return (
        u, z,
        delta_f, B_f, C_f,
        delta_b, B_b, C_b,
        A_f, D_f, delta_bias_f,
        A_b, D_b, delta_bias_b,
    )


def _selective_scan_1d(u_f32, delta_f32, A_f32, B_f32, C_f32, D_f32, delta_bias_f32, reverse=False):
    B_sz, L, D = u_f32.shape
    N = A_f32.shape[1]

    dt = jax.nn.softplus(delta_f32 + delta_bias_f32[None, None, :])

    u_t = jnp.swapaxes(u_f32, 0, 1)
    dt_t = jnp.swapaxes(dt, 0, 1)
    B_t = jnp.swapaxes(B_f32, 0, 1)
    C_t = jnp.swapaxes(C_f32, 0, 1)

    def step_fn(h, xs):
        u_step, dt_step, B_step, C_step = xs
        dA = jnp.exp(dt_step[:, :, None] * A_f32[None, :, :])
        dB_u = (dt_step * u_step)[:, :, None] * B_step[:, None, :]
        h_new = dA * h + dB_u
        y_step = jnp.sum(h_new * C_step[:, None, :], axis=-1) + D_f32[None, :] * u_step
        return h_new, y_step

    h0 = jnp.zeros((B_sz, D, N), dtype=jnp.float32)
    _, y_t = lax.scan(step_fn, h0, (u_t, dt_t, B_t, C_t), reverse=reverse)
    return jnp.swapaxes(y_t, 0, 1)


def workload(
    u, z,
    delta_f, B_f, C_f,
    delta_b, B_b, C_b,
    A_f, D_f, delta_bias_f,
    A_b, D_b, delta_bias_b,
):
    orig_dtype = u.dtype
    u_f32 = u.astype(jnp.float32)
    z_f32 = z.astype(jnp.float32)

    y_f = _selective_scan_1d(
        u_f32,
        delta_f.astype(jnp.float32),
        A_f.astype(jnp.float32),
        B_f.astype(jnp.float32),
        C_f.astype(jnp.float32),
        D_f.astype(jnp.float32),
        delta_bias_f.astype(jnp.float32),
        reverse=False,
    )

    y_b = _selective_scan_1d(
        u_f32,
        delta_b.astype(jnp.float32),
        A_b.astype(jnp.float32),
        B_b.astype(jnp.float32),
        C_b.astype(jnp.float32),
        D_b.astype(jnp.float32),
        delta_bias_b.astype(jnp.float32),
        reverse=True,
    )

    gate = jax.nn.silu(z_f32)
    out = (y_f + y_b) * gate
    return out.astype(orig_dtype)
