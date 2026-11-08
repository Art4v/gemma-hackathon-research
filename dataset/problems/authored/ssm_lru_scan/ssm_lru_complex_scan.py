"""Linear Recurrent Unit (LRU): complex-diagonal input projection, recurrence scan, and output projection."""

import jax
import jax.numpy as jnp
from jax import lax

CONFIG = {
    'name': 'ssm_lru_complex_scan',
    'batch': 16,
    'seq_len': 2048,
    'd_model': 1024,
    'd_state': 1024,
    'r_min': 0.9,
    'r_max': 0.999,
    'max_phase': 0.314159265,
}


def create_inputs(dtype=jnp.bfloat16):
    batch = CONFIG['batch']
    seq_len = CONFIG['seq_len']
    d_model = CONFIG['d_model']
    d_state = CONFIG['d_state']
    r_min = CONFIG['r_min']
    r_max = CONFIG['r_max']
    max_phase = CONFIG['max_phase']

    key = jax.random.key(0)
    k_x, k_u1, k_u2, k_bre, k_bim, k_cre, k_cim, k_d = jax.random.split(key, 8)

    x = jax.random.normal(k_x, (batch, seq_len, d_model), dtype=dtype)

    u1 = jax.random.uniform(k_u1, (d_state,), dtype=jnp.float32)
    u2 = jax.random.uniform(k_u2, (d_state,), dtype=jnp.float32)
    nu_log = jnp.log(-0.5 * jnp.log(u1 * (r_max**2 - r_min**2) + r_min**2))
    theta_log = jnp.log(max_phase * u2)

    r = jnp.exp(-jnp.exp(nu_log))
    theta = jnp.exp(theta_log)
    lambda_re = (r * jnp.cos(theta)).astype(dtype)
    lambda_im = (r * jnp.sin(theta)).astype(dtype)
    gamma_log = jnp.log(jnp.sqrt(1.0 - r**2)).astype(dtype)

    B_re = jax.random.normal(k_bre, (d_model, d_state), dtype=dtype) * 0.02
    B_im = jax.random.normal(k_bim, (d_model, d_state), dtype=dtype) * 0.02
    C_re = jax.random.normal(k_cre, (d_state, d_model), dtype=dtype) * 0.02
    C_im = jax.random.normal(k_cim, (d_state, d_model), dtype=dtype) * 0.02
    D = jnp.ones((d_model,), dtype=dtype)

    return x, lambda_re, lambda_im, B_re, B_im, C_re, C_im, D, gamma_log


def workload(x, lambda_re, lambda_im, B_re, B_im, C_re, C_im, D, gamma_log):
    orig_dtype = x.dtype
    batch = x.shape[0]
    d_state = lambda_re.shape[0]

    gamma = jnp.exp(gamma_log.astype(jnp.float32))
    u_re = jnp.matmul(x, B_re, preferred_element_type=jnp.float32) * gamma
    u_im = jnp.matmul(x, B_im, preferred_element_type=jnp.float32) * gamma

    u_re_t = jnp.swapaxes(u_re, 0, 1)
    u_im_t = jnp.swapaxes(u_im, 0, 1)

    lam_re = lambda_re.astype(jnp.float32)
    lam_im = lambda_im.astype(jnp.float32)

    def step_fn(carry, inputs):
        h_re, h_im = carry
        bu_re, bu_im = inputs
        new_h_re = lam_re * h_re - lam_im * h_im + bu_re
        new_h_im = lam_re * h_im + lam_im * h_re + bu_im
        return (new_h_re, new_h_im), (new_h_re, new_h_im)

    init_re = jnp.zeros((batch, d_state), dtype=jnp.float32)
    init_im = jnp.zeros((batch, d_state), dtype=jnp.float32)

    _, (h_re_t, h_im_t) = lax.scan(step_fn, (init_re, init_im), (u_re_t, u_im_t))

    h_re = jnp.swapaxes(h_re_t, 0, 1).astype(orig_dtype)
    h_im = jnp.swapaxes(h_im_t, 0, 1).astype(orig_dtype)

    y = (
        jnp.matmul(h_re, C_re, preferred_element_type=jnp.float32)
        - jnp.matmul(h_im, C_im, preferred_element_type=jnp.float32)
        + x.astype(jnp.float32) * D.astype(jnp.float32)
    )
    return y.astype(orig_dtype)
