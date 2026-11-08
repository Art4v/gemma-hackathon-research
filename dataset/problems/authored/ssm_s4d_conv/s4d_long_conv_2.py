"""S4D diagonal state space model kernel generation and direct causal long convolution."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 's4d_long_conv',
    'batch_size': 16,
    'seq_len': 2048,
    'd_model': 1024,
    'd_state': 64,
    'dt_min': 0.001,
    'dt_max': 0.1,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_u, k_dt, k_a_real, k_a_imag, k_b_r, k_b_i, k_c_r, k_c_i, k_d = jax.random.split(key, 9)

    B = CONFIG['batch_size']
    L = CONFIG['seq_len']
    H = CONFIG['d_model']
    N = CONFIG['d_state']
    dt_min = CONFIG['dt_min']
    dt_max = CONFIG['dt_max']

    u = jax.random.normal(k_u, (B, L, H), dtype=dtype)

    # Parameterize in float32 for numerical stability of Vandermonde kernel generation
    rand_dt = jax.random.uniform(k_dt, (H,), dtype=jnp.float32)
    log_dt = jnp.log(dt_min) + rand_dt * (jnp.log(dt_max) - jnp.log(dt_min))

    # S4D-Lin initialization: A_real = -0.5, A_imag = pi * n, with small perturbation
    log_A_real = jnp.log(0.5 * jnp.ones((H, N), dtype=jnp.float32)) + 0.01 * jax.random.normal(k_a_real, (H, N), dtype=jnp.float32)
    A_imag = jnp.pi * jnp.broadcast_to(jnp.arange(N, dtype=jnp.float32), (H, N)) + 0.01 * jax.random.normal(k_a_imag, (H, N), dtype=jnp.float32)

    B_real = jax.random.normal(k_b_r, (H, N), dtype=dtype) * 0.02
    B_imag = jax.random.normal(k_b_i, (H, N), dtype=dtype) * 0.02
    C_real = jax.random.normal(k_c_r, (H, N), dtype=dtype) * 0.02
    C_imag = jax.random.normal(k_c_i, (H, N), dtype=dtype) * 0.02
    D = jnp.ones((H,), dtype=dtype)

    return u, log_dt, log_A_real, A_imag, B_real, B_imag, C_real, C_imag, D


def workload(u, log_dt, log_A_real, A_imag, B_real, B_imag, C_real, C_imag, D):
    B, L, H = u.shape
    N = log_A_real.shape[-1]
    out_dtype = u.dtype

    # Step 1: Materialize diagonal SSM parameters in float32
    dt = jnp.exp(log_dt.astype(jnp.float32))  # (H,)
    A_real = -jnp.exp(log_A_real.astype(jnp.float32))  # (H, N)
    A_imag_f32 = A_imag.astype(jnp.float32)  # (H, N)

    dt_A_real = dt[:, None] * A_real  # (H, N)
    dt_A_imag = dt[:, None] * A_imag_f32  # (H, N)

    # Zero-Order Hold (ZOH) discretization for complex diagonal A:
    # C_tilde = C * B * (exp(dt * A) - 1) / A
    exp_dt_A_mag = jnp.exp(dt_A_real)
    exp_dt_A_r = exp_dt_A_mag * jnp.cos(dt_A_imag)
    exp_dt_A_i = exp_dt_A_mag * jnp.sin(dt_A_imag)

    num_r = exp_dt_A_r - 1.0
    num_i = exp_dt_A_i
    denom = A_real * A_real + A_imag_f32 * A_imag_f32
    zoh_r = (num_r * A_real + num_i * A_imag_f32) / denom
    zoh_i = (num_i * A_real - num_r * A_imag_f32) / denom

    Br = B_real.astype(jnp.float32)
    Bi = B_imag.astype(jnp.float32)
    Cr = C_real.astype(jnp.float32)
    Ci = C_imag.astype(jnp.float32)

    CB_r = Cr * Br - Ci * Bi
    CB_i = Cr * Bi + Ci * Br

    # Factor of 2 for conjugate symmetry in S4D complex pairs
    W_r = 2.0 * (CB_r * zoh_r - CB_i * zoh_i)  # (H, N)
    W_i = 2.0 * (CB_r * zoh_i + CB_i * zoh_r)  # (H, N)

    # Step 2: Compute Vandermonde convolution kernel K[h, l] = Re(sum_n W[h, n] * exp(dt_A[h, n] * l))
    l_idx = jnp.arange(L, dtype=jnp.float32)[None, None, :]  # (1, 1, L)
    mag = jnp.exp(dt_A_real[:, :, None] * l_idx)  # (H, N, L)
    phase = dt_A_imag[:, :, None] * l_idx  # (H, N, L)
    v_r = mag * jnp.cos(phase)
    v_i = mag * jnp.sin(phase)

    K = jnp.sum(W_r[:, :, None] * v_r - W_i[:, :, None] * v_i, axis=1)  # (H, L)

    # Step 3: Direct causal 1D depthwise convolution in float32
    # u: (B, L, H) -> (B, H, L)
    u_f32 = jnp.transpose(u.astype(jnp.float32), (0, 2, 1))
    # Reverse kernel along time dimension for standard cross-correlation semantics in conv_general_dilated
    K_rev = jnp.flip(K, axis=-1)[:, None, :]  # (H, 1, L)

    y_conv = lax.conv_general_dilated(
        lhs=u_f32,
        rhs=K_rev,
        window_strides=(1,),
        padding=[(L - 1, 0)],
        dimension_numbers=('NCH', 'OIH', 'NCH'),
        feature_group_count=H,
    )  # (B, H, L)

    y = jnp.transpose(y_conv, (0, 2, 1)) + u_f32.transpose((0, 2, 1)) * D.astype(jnp.float32)[None, None, :]
    return y.astype(out_dtype)
