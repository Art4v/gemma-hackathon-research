"""S4D layer: diagonal complex SSM kernel generation (ZOH Vandermonde) -> FFT-free causal depthwise long convolution + D skip."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 's4d_kernel_direct_causal_conv',
    'batch': 16,
    'seq_len': 2048,
    'd_model': 1024,
    'd_state': 64,          # full state size; stored as d_state // 2 conjugate pairs
    'dt_min': 0.001,
    'dt_max': 0.1,
    'a_real_init': 0.5,     # S4D-Lin: A = -0.5 + i*pi*n
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    B, L, H = CONFIG['batch'], CONFIG['seq_len'], CONFIG['d_model']
    N2 = CONFIG['d_state'] // 2
    key = jax.random.key(0)
    k_x, k_dt, k_cr, k_ci = jax.random.split(key, 4)

    x = jax.random.normal(k_x, (B, L, H), dtype)

    # log_dt uniform in [log dt_min, log dt_max] (S4D init), float32 parameter.
    u = jax.random.uniform(k_dt, (H,), jnp.float32)
    log_dt = (jnp.log(CONFIG['dt_min'])
              + u * (jnp.log(CONFIG['dt_max']) - jnp.log(CONFIG['dt_min'])))

    # S4D-Lin deterministic A init.
    A_log_re = jnp.full((H, N2), jnp.log(CONFIG['a_real_init']), jnp.float32)
    A_im = jnp.broadcast_to(jnp.pi * jnp.arange(N2, dtype=jnp.float32), (H, N2))

    s = CONFIG['weight_scale']
    C_re = jax.random.normal(k_cr, (H, N2), jnp.float32) * s
    C_im = jax.random.normal(k_ci, (H, N2), jnp.float32) * s

    D = jnp.ones((H,), jnp.float32)
    return x, log_dt, A_log_re, A_im, C_re, C_im, D


def _s4d_kernel(log_dt, A_log_re, A_im, C_re, C_im, L):
    """K[h, l] = 2 * Re( sum_n C'[h,n] * exp(dt*A[h,n] * l) ), ZOH discretisation, B = 1."""
    H, N2 = A_im.shape
    dt = jnp.exp(log_dt.astype(jnp.float32))[:, None]                      # (H, 1)
    A = (-jnp.exp(A_log_re.astype(jnp.float32))
         + 1j * A_im.astype(jnp.float32)).astype(jnp.complex64)            # (H, N2)
    C = (C_re.astype(jnp.float32) + 1j * C_im.astype(jnp.float32)).astype(jnp.complex64)
    dtA = dt * A                                                           # (H, N2)
    Cp = C * (jnp.exp(dtA) - 1.0) / A                                      # (H, N2)
    pos = jnp.arange(L, dtype=jnp.float32)[None, :]                        # (1, L)

    def body(n, acc):
        dtA_n = lax.dynamic_slice_in_dim(dtA, n, 1, axis=1)                # (H, 1)
        Cp_n = lax.dynamic_slice_in_dim(Cp, n, 1, axis=1)                  # (H, 1)
        mag = jnp.exp(jnp.real(dtA_n) * pos)                               # (H, L)
        ph = jnp.imag(dtA_n) * pos
        term = mag * (jnp.real(Cp_n) * jnp.cos(ph) - jnp.imag(Cp_n) * jnp.sin(ph))
        return acc + term

    K = lax.fori_loop(0, N2, body, jnp.zeros((H, L), jnp.float32))
    return 2.0 * K                                                         # (H, L)


def workload(x, log_dt, A_log_re, A_im, C_re, C_im, D):
    Bsz, L, H = x.shape
    K = _s4d_kernel(log_dt, A_log_re, A_im, C_re, C_im, L)                 # (H, L) f32

    # Direct causal depthwise conv: y[b,l,h] = sum_{j<=l} K[h,j] x[b,l-j,h].
    # lax conv is cross-correlation, so flip the kernel along length.
    rhs = jnp.flip(K, axis=1).T[:, None, :]                                # (L, 1, H)  WIO
    xf = x.astype(jnp.float32)
    y = lax.conv_general_dilated(
        xf, rhs,
        window_strides=(1,),
        padding=[(L - 1, 0)],
        dimension_numbers=('NWC', 'WIO', 'NWC'),
        feature_group_count=H,
        precision=lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )                                                                      # (B, L, H)
    y = y + D.astype(jnp.float32)[None, None, :] * xf
    return y.astype(x.dtype)
