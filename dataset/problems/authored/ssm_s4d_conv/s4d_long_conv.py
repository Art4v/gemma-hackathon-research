"""S4D layer: diagonal SSM kernel generation (ZOH, Vandermonde) -> causal direct depthwise long convolution -> + D*u skip -> GELU."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 's4d_long_conv',
    'batch': 16,
    'seq_len': 2048,
    'd_model': 1024,
    'd_state': 64,          # N; N/2 complex-conjugate modes are stored
    'dt_min': 0.001,
    'dt_max': 0.1,
    'a_real_init': 0.5,     # S4D-Lin: A = -0.5 + i*pi*n
    'c_init_scale': 0.7071067811865476,  # complex normal with variance 0.5 per component
}


def create_inputs(dtype=jnp.float32):
    B = CONFIG['batch']
    L = CONFIG['seq_len']
    H = CONFIG['d_model']
    N2 = CONFIG['d_state'] // 2
    key = jax.random.key(0)
    k_u, k_dt, k_cr, k_ci, k_d = jax.random.split(key, 5)

    u = jax.random.normal(k_u, (B, L, H), dtype=jnp.float32).astype(dtype)

    # SSM parameters are kept in float32 (phase accuracy over long sequences).
    log_dt = jax.random.uniform(
        k_dt, (H,), dtype=jnp.float32,
        minval=jnp.log(CONFIG['dt_min']), maxval=jnp.log(CONFIG['dt_max']))
    log_A_real = jnp.log(CONFIG['a_real_init'] * jnp.ones((H, N2), dtype=jnp.float32))
    A_imag = jnp.pi * jnp.broadcast_to(
        jnp.arange(N2, dtype=jnp.float32)[None, :], (H, N2))
    C_re = jax.random.normal(k_cr, (H, N2), dtype=jnp.float32) * CONFIG['c_init_scale']
    C_im = jax.random.normal(k_ci, (H, N2), dtype=jnp.float32) * CONFIG['c_init_scale']
    D = jax.random.normal(k_d, (H,), dtype=jnp.float32)
    return (u, log_dt, log_A_real, A_imag, C_re, C_im, D)


def _s4d_kernel(log_dt, log_A_real, A_imag, C_re, C_im, L):
    """K[h, l] = 2 * Re( sum_n C'[h,n] * exp(dt*A[h,n] * l) ), all in float32."""
    dt = jnp.exp(log_dt.astype(jnp.float32))[:, None]          # (H, 1)
    a_re = -jnp.exp(log_A_real.astype(jnp.float32))            # (H, N2)
    a_im = A_imag.astype(jnp.float32)
    c_re = C_re.astype(jnp.float32)
    c_im = C_im.astype(jnp.float32)

    dta_re = dt * a_re
    dta_im = dt * a_im

    # exp(dtA) - 1
    e_mag = jnp.exp(dta_re)
    em1_re = e_mag * jnp.cos(dta_im) - 1.0
    em1_im = e_mag * jnp.sin(dta_im)

    # (exp(dtA) - 1) / A
    denom = a_re * a_re + a_im * a_im
    q_re = (em1_re * a_re + em1_im * a_im) / denom
    q_im = (em1_im * a_re - em1_re * a_im) / denom

    # C' = C * q
    cp_re = c_re * q_re - c_im * q_im                          # (H, N2)
    cp_im = c_re * q_im + c_im * q_re

    # Vandermonde: exp(dtA * l)
    pos = jnp.arange(L, dtype=jnp.float32)                     # (L,)
    mag = jnp.exp(dta_re[:, :, None] * pos[None, None, :])     # (H, N2, L)
    phase = dta_im[:, :, None] * pos[None, None, :]
    v_re = mag * jnp.cos(phase)
    v_im = mag * jnp.sin(phase)

    hp = lax.Precision.HIGHEST
    k = (jnp.einsum('hn,hnl->hl', cp_re, v_re, precision=hp)
         - jnp.einsum('hn,hnl->hl', cp_im, v_im, precision=hp))
    return 2.0 * k                                             # (H, L)


def workload(u, log_dt, log_A_real, A_imag, C_re, C_im, D):
    B, L, H = u.shape
    out_dtype = u.dtype

    K = _s4d_kernel(log_dt, log_A_real, A_imag, C_re, C_im, L)   # (H, L) f32

    # Causal direct depthwise convolution: y[b,l,h] = sum_{j<=l} K[h,j] u[b,l-j,h]
    x = jnp.transpose(u.astype(jnp.float32), (0, 2, 1))          # (B, H, L)
    rhs = jnp.flip(K, axis=-1)[:, None, :]                       # (H, 1, L)
    y = lax.conv_general_dilated(
        x, rhs,
        window_strides=(1,),
        padding=[(L - 1, 0)],
        dimension_numbers=('NCH', 'OIH', 'NCH'),
        feature_group_count=H,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )                                                            # (B, H, L)
    y = jnp.transpose(y, (0, 2, 1))                              # (B, L, H)

    y = y + D.astype(jnp.float32)[None, None, :] * u.astype(jnp.float32)
    y = jax.nn.gelu(y, approximate=False)
    return y.astype(out_dtype)
