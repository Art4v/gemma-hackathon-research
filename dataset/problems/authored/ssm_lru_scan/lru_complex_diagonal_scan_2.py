"""Linear Recurrent Unit: complex B-projection -> complex-diagonal linear recurrence (associative scan) -> Re(C h) + D*x."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'lru_complex_diagonal_scan',
    'batch': 8,
    'seq_len': 4096,
    'd_model': 1024,
    'd_state': 1024,
    'r_min': 0.9,
    'r_max': 0.999,
    'max_phase': 6.283185307179586 / 50.0,
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    b, l, h, n = CONFIG['batch'], CONFIG['seq_len'], CONFIG['d_model'], CONFIG['d_state']
    key = jax.random.key(0)
    kx, kbr, kbi, kcr, kci, kd = jax.random.split(key, 6)
    x = jax.random.normal(kx, (b, l, h), dtype)
    ws = CONFIG['weight_scale']
    B_re = jax.random.normal(kbr, (n, h), dtype) * ws
    B_im = jax.random.normal(kbi, (n, h), dtype) * ws
    C_re = jax.random.normal(kcr, (h, n), dtype) * ws
    C_im = jax.random.normal(kci, (h, n), dtype) * ws
    D = jax.random.normal(kd, (h,), dtype) * ws
    # Deterministic LRU ring initialisation (recurrence parameters kept in float32).
    r_min, r_max = CONFIG['r_min'], CONFIG['r_max']
    u = jnp.linspace(0.0, 1.0, n, dtype=jnp.float32)
    r_sq = u * (r_max ** 2 - r_min ** 2) + r_min ** 2
    nu_log = jnp.log(-0.5 * jnp.log(r_sq))
    phase = jnp.linspace(1.0 / n, 1.0, n, dtype=jnp.float32) * CONFIG['max_phase']
    theta_log = jnp.log(phase)
    gamma_log = jnp.log(jnp.sqrt(1.0 - r_sq))
    return (x, B_re, B_im, C_re, C_im, D, nu_log, theta_log, gamma_log)


def _complex_combine(left, right):
    a1_re, a1_im, b1_re, b1_im = left
    a2_re, a2_im, b2_re, b2_im = right
    a_re = a2_re * a1_re - a2_im * a1_im
    a_im = a2_re * a1_im + a2_im * a1_re
    b_re = a2_re * b1_re - a2_im * b1_im + b2_re
    b_im = a2_re * b1_im + a2_im * b1_re + b2_im
    return a_re, a_im, b_re, b_im


def workload(x, B_re, B_im, C_re, C_im, D, nu_log, theta_log, gamma_log):
    f32 = jnp.float32
    xf = x.astype(f32)
    mag = jnp.exp(-jnp.exp(nu_log.astype(f32)))
    ang = jnp.exp(theta_log.astype(f32))
    lam_re = mag * jnp.cos(ang)
    lam_im = mag * jnp.sin(ang)
    gamma = jnp.exp(gamma_log.astype(f32))

    bu_re = jnp.einsum('blh,nh->bln', x, B_re, preferred_element_type=f32) * gamma
    bu_im = jnp.einsum('blh,nh->bln', x, B_im, preferred_element_type=f32) * gamma

    a_re = jnp.broadcast_to(lam_re, bu_re.shape)
    a_im = jnp.broadcast_to(lam_im, bu_im.shape)
    _, _, h_re, h_im = lax.associative_scan(
        _complex_combine, (a_re, a_im, bu_re, bu_im), axis=1)

    cdt = C_re.dtype
    y = (jnp.einsum('bln,hn->blh', h_re.astype(cdt), C_re, preferred_element_type=f32)
         - jnp.einsum('bln,hn->blh', h_im.astype(cdt), C_im, preferred_element_type=f32))
    y = y + D.astype(f32) * xf
    return y.astype(x.dtype)
