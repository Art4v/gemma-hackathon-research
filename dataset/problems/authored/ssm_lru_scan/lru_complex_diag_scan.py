"""LRU layer: complex-diagonal linear recurrence h_t = lambda*h_{t-1} + (gamma*B) x_t (split re/im), y = Re(C h) + D*x."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'lru_complex_diag_scan',
    'batch': 8,
    'seq_len': 4096,
    'd_model': 1024,
    'd_state': 512,
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
    s = CONFIG['weight_scale']
    B_re = jax.random.normal(kbr, (n, h), dtype) * s
    B_im = jax.random.normal(kbi, (n, h), dtype) * s
    C_re = jax.random.normal(kcr, (h, n), dtype) * s
    C_im = jax.random.normal(kci, (h, n), dtype) * s
    D = jax.random.normal(kd, (h,), dtype) * s
    # Deterministic LRU ring initialization (linspace instead of uniform sampling).
    # Kept in float32: lambda near the unit circle is precision-sensitive.
    r_min, r_max = CONFIG['r_min'], CONFIG['r_max']
    u = (jnp.arange(n, dtype=jnp.float32) + 0.5) / n
    nu_log = jnp.log(-0.5 * jnp.log(u * (r_max ** 2 - r_min ** 2) + r_min ** 2))
    theta_log = jnp.log(CONFIG['max_phase'] * u)
    lam_mod = jnp.exp(-jnp.exp(nu_log))
    gamma_log = jnp.log(jnp.sqrt(1.0 - lam_mod ** 2))
    return x, nu_log, theta_log, gamma_log, B_re, B_im, C_re, C_im, D


def _combine(left, right):
    a1r, a1i, b1r, b1i = left
    a2r, a2i, b2r, b2i = right
    # (a1, b1) then (a2, b2): a = a2*a1, b = a2*b1 + b2 (complex)
    ar = a2r * a1r - a2i * a1i
    ai = a2r * a1i + a2i * a1r
    br = a2r * b1r - a2i * b1i + b2r
    bi = a2r * b1i + a2i * b1r + b2i
    return ar, ai, br, bi


def workload(x, nu_log, theta_log, gamma_log, B_re, B_im, C_re, C_im, D):
    f32 = jnp.float32
    nu = jnp.exp(nu_log.astype(f32))
    theta = jnp.exp(theta_log.astype(f32))
    mod = jnp.exp(-nu)
    lam_re = mod * jnp.cos(theta)
    lam_im = mod * jnp.sin(theta)
    gamma = jnp.exp(gamma_log.astype(f32))

    # Input projection, f32 accumulation; normalize by gamma afterwards.
    bu_re = jnp.einsum('blh,nh->bln', x, B_re, preferred_element_type=f32) * gamma
    bu_im = jnp.einsum('blh,nh->bln', x, B_im, preferred_element_type=f32) * gamma

    a_re = jnp.broadcast_to(lam_re, bu_re.shape)
    a_im = jnp.broadcast_to(lam_im, bu_im.shape)
    _, _, h_re, h_im = lax.associative_scan(
        _combine, (a_re, a_im, bu_re, bu_im), axis=1)

    # y = Re(C h) = C_re h_re - C_im h_im, f32 accumulation.
    y = (jnp.einsum('bln,hn->blh', h_re, C_re.astype(f32), preferred_element_type=f32)
         - jnp.einsum('bln,hn->blh', h_im, C_im.astype(f32), preferred_element_type=f32))
    y = y + D.astype(f32) * x.astype(f32)
    return y.astype(x.dtype)
