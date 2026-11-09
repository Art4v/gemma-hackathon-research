"""[m1 M2_second_output variant of authored/ssm_s4d_conv/s4d_kernel_causal_conv] S4D layer (SSM kernel generation -> causal depthwise long conv -> D skip -> GELU), also returning the per-channel mean over the sequence (the pooled features an S4 classifier head uses)."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 's4d_kernel_causal_conv_m1_aux',
    'batch': 8,
    'seq_len': 4096,
    'd_model': 1024,
    'd_state': 64,          # N; uses N/2 complex-conjugate-paired modes
    'dt_min': 0.001,
    'dt_max': 0.1,
    'a_real_init': 0.5,     # S4D-Lin: A_real = -0.5
    'c_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    B, L, H = CONFIG['batch'], CONFIG['seq_len'], CONFIG['d_model']
    N2 = CONFIG['d_state'] // 2
    key = jax.random.key(0)
    k_u, k_c, k_d = jax.random.split(key, 3)
    u = jax.random.normal(k_u, (B, L, H), dtype=jnp.float32).astype(dtype)
    # Deterministic S4D-Lin initialization for the SSM parameters (kept in float32).
    log_dt = jnp.linspace(jnp.log(CONFIG['dt_min']), jnp.log(CONFIG['dt_max']), H,
                          dtype=jnp.float32)
    log_A_real = jnp.full((H, N2), jnp.log(CONFIG['a_real_init']), dtype=jnp.float32)
    A_imag = jnp.broadcast_to(jnp.pi * jnp.arange(N2, dtype=jnp.float32), (H, N2))
    C = jax.random.normal(k_c, (H, N2, 2), dtype=jnp.float32) * CONFIG['c_scale']
    D = jax.random.normal(k_d, (H,), dtype=jnp.float32)
    return u, log_dt, log_A_real, A_imag, C, D


def _s4d_kernel(log_dt, log_A_real, A_imag, C, L):
    """K[h, l] = 2 * Re(sum_n C'[h,n] * exp(dt*A[h,n] * l)), float32, shape (H, L)."""
    dt = jnp.exp(log_dt.astype(jnp.float32))[:, None]                 # (H, 1)
    A = (-jnp.exp(log_A_real.astype(jnp.float32))
         + 1j * A_imag.astype(jnp.float32)).astype(jnp.complex64)     # (H, N2)
    dtA = A * dt                                                      # (H, N2)
    Cc = (C[..., 0] + 1j * C[..., 1]).astype(jnp.complex64)           # (H, N2)
    Cp = Cc * (jnp.exp(dtA) - 1.0) / A                                # ZOH, B = 1
    re, im = jnp.real(dtA), jnp.imag(dtA)                             # (H, N2) f32
    cr, ci = jnp.real(Cp), jnp.imag(Cp)
    pos = jnp.arange(L, dtype=jnp.float32)[None, :]                   # (1, L)

    def body(K, xs):
        r, i, a, b = xs                                               # each (H,)
        ang = i[:, None] * pos
        term = jnp.exp(r[:, None] * pos) * (a[:, None] * jnp.cos(ang)
                                            - b[:, None] * jnp.sin(ang))
        return K + 2.0 * term, None

    H = log_dt.shape[0]
    K0 = jnp.zeros((H, L), dtype=jnp.float32)
    K, _ = lax.scan(body, K0, (re.T, im.T, cr.T, ci.T))
    return K


def workload(u, log_dt, log_A_real, A_imag, C, D):
    Bsz, L, H = u.shape
    K = _s4d_kernel(log_dt, log_A_real, A_imag, C, L)                 # (H, L) f32
    x = u.astype(jnp.float32)
    # Direct causal depthwise convolution: y[l] = sum_{j<=l} K[j] * x[l-j].
    # conv_general_dilated is cross-correlation, so flip the kernel along time.
    rhs = jnp.flip(K, axis=1).T[:, None, :]                           # (W=L, I=1, O=H)
    y = lax.conv_general_dilated(
        x, rhs,
        window_strides=(1,),
        padding=[(L - 1, 0)],
        dimension_numbers=('NWC', 'WIO', 'NWC'),
        feature_group_count=H,
        precision=lax.Precision.HIGHEST,
    )                                                                 # (B, L, H)
    y = y + x * D.astype(jnp.float32)[None, None, :]
    y = jax.nn.gelu(y, approximate=False)
    pooled = jnp.mean(y, axis=1)                                      # (B, H) f32
    return y.astype(u.dtype), pooled.astype(u.dtype)
