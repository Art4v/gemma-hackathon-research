"""Linear Recurrent Unit (LRU): complex-diagonal recurrence with real/imag split and input/output projections."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ssm_lru_scan',
    'batch_size': 16,
    'seq_len': 2048,
    'd_model': 1024,
    'd_state': 1024,
    'nu_min': 0.001,
    'nu_max': 0.1,
    'theta_max': 0.314159,
    'eps': 1e-6,
}


def create_inputs(dtype=jnp.bfloat16):
    b = CONFIG['batch_size']
    s = CONFIG['seq_len']
    d_model = CONFIG['d_model']
    d_state = CONFIG['d_state']

    k_x, k_br, k_bi, k_cr, k_ci, k_d, k_nu, k_th = jax.random.split(jax.random.key(0), 8)

    x = jax.random.normal(k_x, (b, s, d_model), dtype=dtype)

    b_re = jax.random.normal(k_br, (d_model, d_state), dtype=dtype) * 0.02
    b_im = jax.random.normal(k_bi, (d_model, d_state), dtype=dtype) * 0.02
    c_re = jax.random.normal(k_cr, (d_state, d_model), dtype=dtype) * 0.02
    c_im = jax.random.normal(k_ci, (d_state, d_model), dtype=dtype) * 0.02
    d_proj = jax.random.normal(k_d, (d_model,), dtype=dtype) * 0.02

    u1 = jax.random.uniform(k_nu, (d_state,), dtype=jnp.float32)
    u2 = jax.random.uniform(k_th, (d_state,), dtype=jnp.float32)

    nu_log = jnp.log(-0.5 * jnp.log(u1 * (jnp.exp(-CONFIG['nu_min']) - jnp.exp(-CONFIG['nu_max'])) + jnp.exp(-CONFIG['nu_max'])))
    theta_log = jnp.log(u2 * CONFIG['theta_max'])
    gamma_log = jnp.log(jnp.sqrt(jnp.maximum(1.0 - jnp.exp(-2.0 * jnp.exp(nu_log)), CONFIG['eps'])))

    return x, b_re, b_im, c_re, c_im, d_proj, nu_log.astype(dtype), theta_log.astype(dtype), gamma_log.astype(dtype)


def workload(x, b_re, b_im, c_re, c_im, d_proj, nu_log, theta_log, gamma_log):
    orig_dtype = x.dtype

    nu = jnp.exp(nu_log.astype(jnp.float32))
    theta = jnp.exp(theta_log.astype(jnp.float32))
    gamma = jnp.exp(gamma_log.astype(jnp.float32))

    lam_re = jnp.exp(-nu) * jnp.cos(theta)
    lam_im = jnp.exp(-nu) * jnp.sin(theta)

    u_re = jnp.matmul(x, b_re, preferred_element_type=jnp.float32) * gamma
    u_im = jnp.matmul(x, b_im, preferred_element_type=jnp.float32) * gamma

    s = x.shape[1]
    a_re = jnp.broadcast_to(lam_re[None, None, :], u_re.shape)
    a_im = jnp.broadcast_to(lam_im[None, None, :], u_im.shape)

    def complex_assoc_op(left, right):
        ar_l, ai_l, br_l, bi_l = left
        ar_r, ai_r, br_r, bi_r = right
        ar_new = ar_r * ar_l - ai_r * ai_l
        ai_new = ar_r * ai_l + ai_r * ar_l
        br_new = ar_r * br_l - ai_r * bi_l + br_r
        bi_new = ar_r * bi_l + ai_r * br_l + bi_r
        return ar_new, ai_new, br_new, bi_new

    _, _, h_re, h_im = lax.associative_scan(complex_assoc_op, (a_re, a_im, u_re, u_im), axis=1)

    h_re_cast = h_re.astype(orig_dtype)
    h_im_cast = h_im.astype(orig_dtype)

    y_re = (
        jnp.matmul(h_re_cast, c_re, preferred_element_type=jnp.float32)
        - jnp.matmul(h_im_cast, c_im, preferred_element_type=jnp.float32)
    )
    y = y_re + x.astype(jnp.float32) * d_proj.astype(jnp.float32)
    return y.astype(orig_dtype)
