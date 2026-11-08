"""Linear Recurrent Unit (LRU) complex-diagonal recurrence and projections."""
import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ssm_lru_scan',
    'batch_size': 16,
    'seq_len': 2048,
    'd_model': 1024,
    'd_state': 512,
    'r_min': 0.9,
    'r_max': 0.999,
    'max_phase': 0.314159265,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k_x, k_nu, k_th, k_br, k_bi, k_cr, k_ci, k_d, k_g = jax.random.split(key, 9)
    B = CONFIG['batch_size']
    L = CONFIG['seq_len']
    D = CONFIG['d_model']
    N = CONFIG['d_state']

    x = jax.random.normal(k_x, (B, L, D), dtype=dtype)

    u1 = jax.random.uniform(k_nu, (N,), dtype=jnp.float32)
    r_min = CONFIG['r_min']
    r_max = CONFIG['r_max']
    nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))

    u2 = jax.random.uniform(k_th, (N,), dtype=jnp.float32)
    theta_log = jnp.log(u2 * CONFIG['max_phase'])

    B_re = jax.random.normal(k_br, (D, N), dtype=dtype) * 0.02
    B_im = jax.random.normal(k_bi, (D, N), dtype=dtype) * 0.02
    C_re = jax.random.normal(k_cr, (N, D), dtype=dtype) * 0.02
    C_im = jax.random.normal(k_ci, (N, D), dtype=dtype) * 0.02
    D_param = jax.random.normal(k_d, (D,), dtype=dtype) * 0.02

    lam_abs = jnp.exp(-jnp.exp(nu_log))
    gamma_log = jnp.log(jnp.sqrt(1.0 - lam_abs ** 2))

    return x, nu_log, theta_log, B_re, B_im, C_re, C_im, D_param, gamma_log


def workload(x, nu_log, theta_log, B_re, B_im, C_re, C_im, D_param, gamma_log):
    orig_dtype = x.dtype
    B_sz, L, _ = x.shape
    N = nu_log.shape[0]

    nu = jnp.exp(nu_log.astype(jnp.float32))
    theta = jnp.exp(theta_log.astype(jnp.float32))
    lam_mag = jnp.exp(-nu)
    lam_re = lam_mag * jnp.cos(theta)
    lam_im = lam_mag * jnp.sin(theta)
    gamma = jnp.exp(gamma_log.astype(jnp.float32))

    u_re = jnp.einsum('bld,dn->bln', x, B_re, preferred_element_type=jnp.float32) * gamma
    u_im = jnp.einsum('bld,dn->bln', x, B_im, preferred_element_type=jnp.float32) * gamma

    a_re = jnp.broadcast_to(lam_re[None, None, :], (B_sz, L, N))
    a_im = jnp.broadcast_to(lam_im[None, None, :], (B_sz, L, N))

    def binary_op(left, right):
        ar1, ai1, br1, bi1 = left
        ar2, ai2, br2, bi2 = right
        ar_out = ar2 * ar1 - ai2 * ai1
        ai_out = ar2 * ai1 + ai2 * ar1
        br_out = ar2 * br1 - ai2 * bi1 + br2
        bi_out = ar2 * bi1 + ai2 * br1 + bi2
        return ar_out, ai_out, br_out, bi_out

    _, _, h_re, h_im = lax.associative_scan(binary_op, (a_re, a_im, u_re, u_im), axis=1)

    h_re_cast = h_re.astype(orig_dtype)
    h_im_cast = h_im.astype(orig_dtype)

    y_re = (
        jnp.einsum('bln,nd->bld', h_re_cast, C_re, preferred_element_type=jnp.float32)
        - jnp.einsum('bln,nd->bld', h_im_cast, C_im, preferred_element_type=jnp.float32)
    )
    y = y_re + x.astype(jnp.float32) * D_param.astype(jnp.float32)[None, None, :]
    return y.astype(orig_dtype)
