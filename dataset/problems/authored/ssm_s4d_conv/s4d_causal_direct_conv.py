"""S4D diagonal state space model kernel generation and direct causal 1D convolution."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 's4d_causal_direct_conv',
    'batch_size': 16,
    'seq_len': 2048,
    'd_model': 1024,
    'd_state': 64,
    'dt_min': 0.001,
    'dt_max': 0.1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_u, k_ar, k_ai, k_cr, k_ci, k_dt, k_d = jax.random.split(key, 7)

    B = CONFIG['batch_size']
    L = CONFIG['seq_len']
    H = CONFIG['d_model']
    N = CONFIG['d_state']

    u = jax.random.normal(k_u, (B, L, H), dtype=dtype)

    log_A_real = jax.random.uniform(
        k_ar, (H, N // 2), dtype=jnp.float32, minval=-1.0, maxval=-0.1
    ).astype(dtype)
    A_imag = (
        jnp.pi
        * jnp.broadcast_to(
            jnp.arange(N // 2, dtype=jnp.float32)[None, :], (H, N // 2)
        )
        + jax.random.normal(k_ai, (H, N // 2), dtype=jnp.float32) * 0.02
    ).astype(dtype)

    C_real = jax.random.normal(k_cr, (H, N // 2), dtype=dtype) * 0.02
    C_imag = jax.random.normal(k_ci, (H, N // 2), dtype=dtype) * 0.02

    log_dt_min = jnp.log(CONFIG['dt_min'])
    log_dt_max = jnp.log(CONFIG['dt_max'])
    log_dt = jax.random.uniform(
        k_dt, (H,), dtype=jnp.float32, minval=log_dt_min, maxval=log_dt_max
    ).astype(dtype)

    D = jax.random.normal(k_d, (H,), dtype=dtype) * 0.02

    return u, log_A_real, A_imag, C_real, C_imag, log_dt, D


def workload(u, log_A_real, A_imag, C_real, C_imag, log_dt, D):
    B, L, H = u.shape
    orig_dtype = u.dtype

    u_f32 = u.astype(jnp.float32)
    dt = jnp.exp(log_dt.astype(jnp.float32))
    A_real = -jnp.exp(log_A_real.astype(jnp.float32))
    A_im = A_imag.astype(jnp.float32)
    C_r = C_real.astype(jnp.float32)
    C_i = C_imag.astype(jnp.float32)
    D_f32 = D.astype(jnp.float32)

    dt_A_real = dt[:, None] * A_real
    dt_A_imag = dt[:, None] * A_im

    exp_dtA_mag = jnp.exp(dt_A_real)
    dB_real = exp_dtA_mag * jnp.cos(dt_A_imag) - 1.0
    dB_imag = exp_dtA_mag * jnp.sin(dt_A_imag)

    denom = A_real * A_real + A_im * A_im
    B_bar_real = (dB_real * A_real + dB_imag * A_im) / denom
    B_bar_imag = (dB_imag * A_real - dB_real * A_im) / denom

    W_real = 2.0 * (C_r * B_bar_real - C_i * B_bar_imag)
    W_imag = 2.0 * (C_r * B_bar_imag + C_i * B_bar_real)

    t = jnp.arange(L, dtype=jnp.float32)[None, None, :]
    r_t = jnp.exp(dt_A_real[:, :, None] * t)
    theta_t = dt_A_imag[:, :, None] * t

    K = jnp.sum(
        W_real[:, :, None] * r_t * jnp.cos(theta_t)
        - W_imag[:, :, None] * r_t * jnp.sin(theta_t),
        axis=1,
    )

    u_t = jnp.transpose(u_f32, (0, 2, 1))
    K_rev = jnp.flip(K, axis=-1)[:, None, :]

    y_t = lax.conv_general_dilated(
        lhs=u_t,
        rhs=K_rev,
        window_strides=(1,),
        padding=((L - 1, 0),),
        feature_group_count=H,
        dimension_numbers=('NCH', 'OIH', 'NCH'),
    )

    y = jnp.transpose(y_t, (0, 2, 1)) + u_f32 * D_f32[None, None, :]
    return y.astype(orig_dtype)
