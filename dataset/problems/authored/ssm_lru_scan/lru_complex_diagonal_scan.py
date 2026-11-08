"""Linear Recurrent Unit: complex input projection -> complex-diagonal associative scan (re/im split) -> Re(C h) + D x."""
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
    'max_phase': 0.3141592653589793,  # pi / 10
    'weight_scale': 0.02,
}


def create_inputs(dtype=jnp.float32):
    B, L, H, N = CONFIG['batch'], CONFIG['seq_len'], CONFIG['d_model'], CONFIG['d_state']
    keys = jax.random.split(jax.random.key(0), 8)
    x = jax.random.normal(keys[0], (B, L, H), dtype)

    # Ring init of the complex eigenvalues (kept in float32 for precision near |lambda| ~ 1).
    r_min, r_max = CONFIG['r_min'], CONFIG['r_max']
    u1 = jax.random.uniform(keys[1], (N,), jnp.float32)
    u2 = jax.random.uniform(keys[2], (N,), jnp.float32)
    nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
    theta_log = jnp.log(CONFIG['max_phase'] * u2)
    lam_abs = jnp.exp(-jnp.exp(nu_log))
    gamma_log = jnp.log(jnp.sqrt(1.0 - lam_abs ** 2))

    s = CONFIG['weight_scale']
    B_re = jax.random.normal(keys[3], (N, H), dtype) * s
    B_im = jax.random.normal(keys[4], (N, H), dtype) * s
    C_re = jax.random.normal(keys[5], (H, N), dtype) * s
    C_im = jax.random.normal(keys[6], (H, N), dtype) * s
    D = jnp.ones((H,), dtype)
    return (x, nu_log, theta_log, gamma_log, B_re, B_im, C_re, C_im, D)


def _complex_scan_op(left, right):
    a_re_i, a_im_i, b_re_i, b_im_i = left
    a_re_j, a_im_j, b_re_j, b_im_j = right
    a_re = a_re_j * a_re_i - a_im_j * a_im_i
    a_im = a_re_j * a_im_i + a_im_j * a_re_i
    b_re = a_re_j * b_re_i - a_im_j * b_im_i + b_re_j
    b_im = a_re_j * b_im_i + a_im_j * b_re_i + b_im_j
    return a_re, a_im, b_re, b_im


def workload(x, nu_log, theta_log, gamma_log, B_re, B_im, C_re, C_im, D):
    f32 = jnp.float32
    nu_log = nu_log.astype(f32)
    theta_log = theta_log.astype(f32)
    gamma = jnp.exp(gamma_log.astype(f32))

    mag = jnp.exp(-jnp.exp(nu_log))
    phase = jnp.exp(theta_log)
    lam_re = mag * jnp.cos(phase)
    lam_im = mag * jnp.sin(phase)

    # Complex input projection, fp32 accumulation, normalised by gamma.
    bu_re = jnp.einsum('blh,nh->bln', x, B_re, preferred_element_type=f32) * gamma
    bu_im = jnp.einsum('blh,nh->bln', x, B_im, preferred_element_type=f32) * gamma

    a_re = jnp.broadcast_to(lam_re, bu_re.shape)
    a_im = jnp.broadcast_to(lam_im, bu_im.shape)

    _, _, h_re, h_im = lax.associative_scan(
        _complex_scan_op, (a_re, a_im, bu_re, bu_im), axis=1)

    # Re(C h) = C_re h_re - C_im h_im
    y = (jnp.einsum('bln,hn->blh', h_re, C_re.astype(f32), preferred_element_type=f32)
         - jnp.einsum('bln,hn->blh', h_im, C_im.astype(f32), preferred_element_type=f32))
    y = y + D.astype(f32) * x.astype(f32)
    return y.astype(x.dtype)
